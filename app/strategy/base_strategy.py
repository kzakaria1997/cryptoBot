from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class BaseStrategy(ABC):
    def __init__(self, name: str, params: dict[str, Any] | None = None):
        self.name = name
        self.params = params or {}

    def get_name(self) -> str:
        return self.name

    def get_params(self) -> dict[str, Any]:
        return self.params

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Permet à la stratégie d'ajouter ou valider ses colonnes nécessaires.
        Par défaut, on retourne le DataFrame tel quel.
        """
        return df

    @abstractmethod
    def required_columns(self) -> list[str]:
        """
        Retourne la liste des colonnes nécessaires à la stratégie.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_entry_signal(self, row: pd.Series) -> bool:
        """
        Retourne True si on veut entrer en position sur cette bougie.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_exit_signal(self, row: pd.Series) -> bool:
        """
        Retourne True si on veut sortir sur signal.
        Si tu veux une stratégie sans sortie signal, retourne False.
        """
        raise NotImplementedError

    def validate_row(self, row: pd.Series) -> bool:
        """
        Vérifie que la ligne contient bien toutes les colonnes nécessaires
        et qu'elles ne sont pas NaN.
        """
        cols = self.required_columns()
        if any(col not in row.index for col in cols):
            return False
        return not row[cols].isna().any()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, params={self.params})"