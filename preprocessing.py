"""
PV Data Preprocessing and Feature Engineering Utilities.

Provides data cleaning, multi-source ingestion alignment, temporal interpolation,
and cross-dataset feature syncing logic for solar PV modeling.
"""

from typing import Optional
import numpy as np
import pandas as pd


class PVPreprocessor:
    """
    Orchestration layer responsible for cleansing and merging SCADA data with weather data.

    Attributes:
        pv_data (Optional[pd.DataFrame]): Cleansed PV production timeseries.
        weather_data (Optional[pd.DataFrame]): Cleansed atmospheric readings.
        merged_data (Optional[pd.DataFrame]): Aligned physics features frame.
    """

    pv_data: Optional[pd.DataFrame]
    weather_data: Optional[pd.DataFrame]
    merged_data: Optional[pd.DataFrame]

    def __init__(self) -> None:
        """Initialize empty components for preprocessor pipeline storage."""
        self.pv_data = None
        self.weather_data = None
        self.merged_data = None

    def load_data(self, pv_path: str, weather_path: str) -> None:
        """
        Load raw data sources from local persistent storage files.

        Args:
            pv_path: Path string mapping to raw CSV production data.
            weather_path: Path string mapping to raw Excel weather observations.
        """
        try:
            self.pv_data = pd.read_csv(pv_path)
            self.weather_data = pd.read_excel(weather_path, engine='openpyxl')
            print("Files loaded successfully.")
        except IOError as e:
            print(f"Error loading files via system IO: {e}")

    def clean_pv_data(self) -> None:
        """Clean production metrics and normalise datetime objects."""
        df = self.pv_data.copy()
        df['timestamp'] = pd.to_datetime(df['datetime'], errors='coerce')
        target_col = 'P_GEN'
        df.dropna(subset=['timestamp', target_col], inplace=True)
        # Sort for merging, but DO NOT drop duplicates here to preserve all Substations
        df.sort_values('timestamp', inplace=True)
        self.pv_data = df

    def clean_weather_data(self) -> None:
        """Clean atmospheric records and standardise observation metadata names."""
        if self.weather_data is None:
            return
        df = self.weather_data.copy()

        # Standardize the Substation column name
        if 'Site' in df.columns:
            df.rename(columns={'Site': 'Substation'}, inplace=True)

        df['timestamp_str'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
        df['timestamp'] = pd.to_datetime(df['timestamp_str'], errors='coerce')

        # 'Substation' must be in the final weather dataframe
        cols_to_keep = [
            'timestamp', 'Substation', 'SolarRad', 'SolarEnergy', 'HiSolarRad',
            'TempOut', 'OutHum', 'WindSpeed', 'Rain', 'RainRate',
            'UV', 'Bar', 'DewPt', 'WindDir', 'HiDir'
        ]

        existing_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[existing_cols]
        df.dropna(subset=['timestamp'], inplace=True)
        df.sort_values('timestamp', inplace=True)

        # Drop duplicates based on BOTH timestamp and Substation
        if 'Substation' in df.columns:
            df = df[~df.duplicated(subset=['timestamp', 'Substation'], keep='first')]
        else:
            df = df[~df['timestamp'].duplicated(keep='first')]

        self.weather_data = df

    def merge_datasets(self, tolerance: str = '10min') -> pd.DataFrame:
        """
        Fix timeseries with closest corresponding atmospheric observations.

        Args:
            tolerance: Maximum valid time delta gap allowed between matches.

        Returns:
            pd.DataFrame: Merged and aligned structural records matrix.
        """
        # Both datasets must be sorted by timestamp
        self.pv_data.sort_values('timestamp', inplace=True)
        self.weather_data.sort_values('timestamp', inplace=True)

        # Only merge if the Substation match
        if 'Substation' in self.pv_data.columns and 'Substation' in self.weather_data.columns:
            self.merged_data = pd.merge_asof(
                self.pv_data,
                self.weather_data,
                on='timestamp',
                by='Substation',
                direction='nearest',
                tolerance=pd.Timedelta(tolerance)
            )
        else:
            self.merged_data = pd.merge_asof(
                self.pv_data,
                self.weather_data,
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta(tolerance)
            )

        self.merged_data.dropna(subset=['SolarRad'], inplace=True)
        return self.merged_data

    def encode_categoricals(self) -> None:
        """Encode textual direction maps and location classifications into numeric scales."""
        if self.merged_data is None:
            return
        df = self.merged_data.copy()

        dir_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
            'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225,
            'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }

        for col in ['WindDir', 'HiDir']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().map(dir_map)

        if 'Substation' in df.columns:
            df['Substation_ID'] = df['Substation'].astype('category').cat.codes

        self.merged_data = df

    def handle_nulls(self, inplace: bool = True) -> pd.DataFrame:
        """
        Synchronise dynamic solar physics drift and interpolate null elements.

        Args:
            inplace: Updates current internal pipeline matrix if True.

        Returns:
            pd.DataFrame: Cleaned data tracking physics-aligned values.
        """
        df = self.merged_data.copy()

        weather_cols = [
            'SolarRad', 'SolarEnergy', 'HiSolarRad', 'TempOut',
            'OutHum', 'WindSpeed', 'Rain', 'RainRate', 'UV', 'Bar', 'DewPt'
        ]
        available_weather = [col for col in weather_cols if col in df.columns]

        for col in available_weather:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate shift on the first substation only to prevent cross-contamination
        test_df = df.copy()
        if 'Substation_ID' in test_df.columns:
            first_sub = test_df['Substation_ID'].iloc[0]
            test_df = test_df[test_df['Substation_ID'] == first_sub]

        best_corr, best_shift = -1, 0
        for shift in range(-50, 50):
            corr = test_df['P_GEN'].corr(test_df['SolarRad'].shift(shift))
            if corr > best_corr:
                best_corr, best_shift = corr, shift

        # Shift features grouped by Substation to preserve physics
        if 'Substation_ID' in df.columns:
            for col in available_weather:
                df[col] = df.groupby('Substation_ID')[col].shift(best_shift)
        else:
            for col in available_weather:
                df[col] = df[col].shift(best_shift)

        # Set index for modeling
        df.set_index('timestamp', inplace=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_with_nulls = [col for col in numeric_cols if df[col].isnull().any()]

        if cols_with_nulls:
            # Interpolate per Substation to prevent data bleeding
            if 'Substation_ID' in df.columns:
                transformer = df.groupby('Substation_ID')[cols_with_nulls]
                df[cols_with_nulls] = transformer.transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
            else:
                df[cols_with_nulls] = df[cols_with_nulls].interpolate(
                    method='linear', limit_direction='both'
                )

        features_to_check = ['SolarRad', 'TempOut', 'WindSpeed', 'OutHum', 'P_GEN']
        existing_check = [c for c in features_to_check if c in df.columns]

        df.dropna(subset=existing_check, inplace=True)
        df.dropna(inplace=True)

        if inplace:
            self.merged_data = df

        return df