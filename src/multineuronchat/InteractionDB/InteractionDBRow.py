import pandas as pd

from typing import List, Dict, Set


class InteractionDBRow:
    def __init__(self, row_series: pd.Series):
        self.interaction_name: str = row_series['interaction_name']
        self.ligand_type: str = row_series['ligand_type']
        self.interaction_type: str = row_series['interaction_type']

        self.ligand_contributor: List[str] = row_series['lig_contributor'].split('-')
        self.ligand_contributor_group: List[int] = list(
            map(lambda x: int(x), row_series['lig_contributor_group'].split('-'))
        )
        self.ligand_contributor_coeff: List[int] = list(
            map(lambda x: int(x), row_series['lig_contributor_coeff'].split('-'))
        )

        self.ligand_group_to_gene_dict: Dict[int, List[str]] = {}
        self.ligand_group_to_coeff_dict: Dict[int, int] = {}
        for ligand, group in zip(self.ligand_contributor,
                                 self.ligand_contributor_group):
            if group not in self.ligand_group_to_gene_dict.keys():
                self.ligand_group_to_gene_dict[group] = []
                self.ligand_group_to_coeff_dict[group] = self.ligand_contributor_coeff[group-1]

            self.ligand_group_to_gene_dict[group].append(ligand)

        self.target_subunit: List[str] = row_series['target_subunit'].split('-')
        self.target_subunit_group: List[int] = list(
            map(lambda x: int(x), row_series['target_subunit_group'].split('-'))
        )
        self.target_subunit_coeff: List[int] = list(
            map(lambda x: int(x), row_series['target_subunit_coeff'].split('-'))
        )

        self.target_group_to_gene_dict: Dict[int, List[str]] = {}
        self.target_group_to_coeff_dict: Dict[int, int] = {}
        for target, group in zip(self.target_subunit,
                                 self.target_subunit_group):
            if group not in self.target_group_to_gene_dict.keys():
                self.target_group_to_gene_dict[group] = []
                self.target_group_to_coeff_dict[group] = self.target_subunit_coeff[group-1]

            self.target_group_to_gene_dict[group].append(target)

        # Create sets of groups so that one can simply iterate over them
        self.ligand_groups: Set[int] = set(self.ligand_contributor_group)
        self.target_groups: Set[int] = set(self.target_subunit_group)
