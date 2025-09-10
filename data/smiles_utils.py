"""
Based on https://github.com/BenevolentAI/guacamol/tree/master/guacamol/utils
"""
from typing import List, Iterable, Optional, Set, Any

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem

# Mute RDKit logger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

def filter_SMILES_with_known_tokens(smiles_list, vocabulary):
    mask = list()

    for smiles in smiles_list:
        try:
            vocabulary.encode(smiles)
            mask.append(True)
        except KeyError:
            mask.append(False)

    return mask

class AllowedSmilesCharDictionary(object):
    """
    A fixed dictionary for druglike SMILES.
    """

    def __init__(self, forbidden_symbols: Optional[Set[str]] = None) -> None:
        if forbidden_symbols is None:
            forbidden_symbols = {'Ag', 'Al', 'Am', 'Ar', 'At', 'Au', 'D', 'E', 'Fe', 'G', 'K', 'L', 'M', 'Ra', 'Re',
                                 'Rf', 'Rg', 'Rh', 'Ru', 'T', 'U', 'V', 'W', 'Xe',
                                 'Y', 'Zr', 'a', 'd', 'f', 'g', 'h', 'k', 'm', 'si', 't', 'te', 'u', 'v', 'y'}
        self.forbidden_symbols = forbidden_symbols

    def allowed(self, smiles: str) -> bool:
        """
        Determine if SMILES string has illegal symbols

        Args:
            smiles: SMILES string

        Returns:
            True if all legal
        """
        for symbol in self.forbidden_symbols:
            if symbol in smiles:
                print('Forbidden symbol {:<2}  in  {}'.format(symbol, smiles))
                return False
        return True

def split_charged_mol(smiles: str) -> str:
    if smiles.count('.') > 0:
        largest = ''
        largest_len = -1
        split = smiles.split('.')
        for i in split:
            if len(i) > largest_len:
                largest = i
                largest_len = len(i)
        return largest

    else:
        return smiles

def is_valid(smiles: str):
    """
    Verifies whether a SMILES string corresponds to a valid molecule.

    Args:
        smiles: SMILES string

    Returns:
        True if the SMILES strings corresponds to a valid, non-empty molecule.
    """

    mol = Chem.MolFromSmiles(smiles)

    return smiles != '' and mol is not None and mol.GetNumAtoms() > 0

def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """
    Canonicalize the SMILES strings with RDKit.

    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543

    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string

    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None


def canonicalize_list(smiles_list: Iterable[str], include_stereocenters=True) -> List[str]:
    """
    Canonicalize a list of smiles. Filters out repetitions and removes corrupted molecules.

    Args:
        smiles_list: molecules as SMILES strings
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES strings

    Returns:
        The canonicalized and filtered input smiles.
    """

    canonicalized_smiles = [canonicalize(smiles, include_stereocenters) for smiles in smiles_list]

    # Remove None elements
    canonicalized_smiles = [s for s in canonicalized_smiles if s is not None]

    return remove_duplicates(canonicalized_smiles)

def remove_duplicates(list_with_duplicates):
    """
    Removes the duplicates and keeps the ordering of the original list.
    For duplicates, the first occurrence is kept and the later occurrences are ignored.

    Args:
        list_with_duplicates: list that possibly contains duplicates

    Returns:
        A list with no duplicates.
    """

    unique_set: Set[Any] = set()
    unique_list = []
    for element in list_with_duplicates:
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)

    return unique_list


def neutralise_charges(mol, reactions=None):
    replaced = False

    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        Chem.SanitizeMol(mol)
        return mol, True
    else:
        return mol, False

def initialise_neutralisation_reactions():
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]

def filter_and_canonicalize(smiles: str, neutralization_rxns=None,
                            include_stereocenters=False):
    """
    Args:
        smiles: the molecule to process
        holdout_set: smiles of the holdout set
        holdout_fps: ECFP4 fingerprints of the holdout set
        neutralization_rxns: neutralization rdkit reactions
        tanimoto_cutoff: Remove molecules with a higher ECFP4 tanimoto similarity than this cutoff from the set
        include_stereocenters: whether to keep stereocenters during canonicalization

    Returns:
        list with canonical smiles as a list with one element, or a an empty list. This is to perform a flatmap:
    """
    if neutralization_rxns is None:
        neutralization_rxns = initialise_neutralisation_reactions()

    try:
        # Drop out if too long
        if len(smiles) > 200:
            return []
        mol = Chem.MolFromSmiles(smiles)
        # Drop out if invalid
        if mol is None:
            return []
        mol = Chem.RemoveHs(mol)

        # We only accept molecules consisting of H, B, C, N, O, F, Si, P, S, Cl, aliphatic Se, Br, I.
        metal_smarts = Chem.MolFromSmarts('[!#1!#5!#6!#7!#8!#9!#14!#15!#16!#17!#34!#35!#53]')

        has_metal = mol.HasSubstructMatch(metal_smarts)

        # Exclude molecules containing the forbidden elements.
        if has_metal:
            print(f'metal {smiles}')
            return []

        canon_smi = Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)

        # Drop out if too long canonicalized:
        if len(canon_smi) > 100:
            return []
        # Balance charges if unbalanced
        if canon_smi.count('+') - canon_smi.count('-') != 0:
            new_mol, changed = neutralise_charges(mol, reactions=neutralization_rxns)
            if changed:
                mol = new_mol
                canon_smi = Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)

        return [canon_smi]
    except Exception as e:
        print(e)
    return []

smiles_char_dict = AllowedSmilesCharDictionary()

def clean_smile(smiles):
    # only keep reasonably sized molecules
    if 5 > len(smiles) or len(smiles) > 200:
        return None
    
    smiles = split_charged_mol(smiles)

    if not smiles_char_dict.allowed(smiles):
        return None
    
    filtered_smiles = filter_and_canonicalize(smiles)
    return filtered_smiles[0] if filtered_smiles else None
