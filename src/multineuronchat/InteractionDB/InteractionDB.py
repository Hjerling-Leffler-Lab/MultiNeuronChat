import os.path

from importlib.resources import files

import pandas as pd

from .InteractionDBRow import InteractionDBRow

from typing import Optional, List, Set, Union


class InteractionDB:
    def __init__(self,
                 db: Optional[str] = 'human_extended'):
        if (db not in ['human', 'mouse', 'human_extended']) and (not os.path.exists(db)):
            raise ValueError('DB must be "human", "human_extended", "mouse", or a valid path to a dataset')

        if db in ['human', 'mouse', 'human_extended']:
            with files('multineuronchat.db').joinpath(f'../db/interactionDB_{db}.pkl').open('rb') as file:
            #with files('MultiNeuronChat.db').joinpath(f'../db/interactionDB_{db}.pkl').open('rb') as file:
                self.interaction_df: pd.DataFrame = pd.read_pickle(file)
        else:
            self.interaction_df: pd.DataFrame = pd.read_pickle(db)

        self.__set_of_genes: Set[str] = self.__extract_set_of_genes()

    def __extract_set_of_genes(self) -> Set[str]:
        set_of_genes: Set[str] = set()

        for lig_contributor in self.interaction_df['lig_contributor']:
            genes: List[str] = lig_contributor.split('-')
            set_of_genes.update(genes)

        for target_contributor in self.interaction_df['target_subunit']:
            genes: List[str] = target_contributor.split('-')
            set_of_genes.update(genes)

        return set_of_genes

    def get_set_of_genes(self) -> Set[str]:
        return self.__set_of_genes

    def get_interaction_names(self) -> List[str]:
        return self.interaction_df['interaction_name'].tolist()

    def __getitem__(self, item: Union[str, int]) -> InteractionDBRow:
        if type(item) is str:
            return InteractionDBRow(self.interaction_df[self.interaction_df['interaction_name'] == item].iloc[0])
        elif type(item) is int:
            return InteractionDBRow(self.interaction_df.iloc[item])
        else:
            raise ValueError('Access is only defined for the type str or int')

    def __len__(self) -> int:
        return len(self.interaction_df)

    def __iter__(self):
        self.__iter_idx = 0
        return self

    def __next__(self) -> InteractionDBRow:
        if self.__iter_idx < len(self.interaction_df):
            interaction = InteractionDBRow(self.interaction_df.iloc[self.__iter_idx])
            self.__iter_idx += 1
            return interaction
        else:
            raise StopIteration
