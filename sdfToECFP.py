## assumes we have rdkit installed
## and that chembl has been installed

## About fingerprints
## http://www.chemaxon.com/jchem/doc/user/ECFP.html

from rdkit import Chem
from rdkit.Chem import AllChem
from os.path import expanduser
import argparse
import csv
import sys
import pandas as pd

def defaultSupplFile():
  return '%s/chembl_19/chembl_19.sdf' % datadir

def saveFingerprints(results, filename):
  """ results is a dictionary: chembl_id -> [fingerprints] """
  with open(filename, 'w') as csvfile:
    fpwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_NONE)
    fpwriter.writerow(["compound","feature"])
    for compound in results:
        for feature in results[compound]:
            fpwriter.writerow( [compound, feature] )
       

def getChemblIDs(ic50file):
  a = pd.read_csv(ic50file, header = None)
  chembl = a[0].values 
  return chembl

class SDF:
  def __init__(self, supplFile):
    self.supplFile = supplFile
    self.suppl = Chem.SDMolSupplier( self.supplFile )

  def getMol(self, n = 10e+9):
    """ function for looping over all molecules """
    self.suppl.reset()
    i = 0
    for rdmol in self.suppl:
      if rdmol is None: continue
      i += 1
      yield rdmol
      if i >= n: return

  def print4Mol(self):
    for mol in self.getMol(4):
      bits = AllChem.GetMorganFingerprint(mol, 3) 
      print ('Chembl-id: %s' % mol.GetProp('chembl_id'))
      print ("#bits:     %d" % len(bits.GetNonzeroElements()))
      print ('Smiles:    %s' % Chem.MolToSmiles(mol, isomericSmiles=True))
      print ("")

  def getMorganFingerPrints(self, chemblIDs, nMorgan):
    ids = set(chemblIDs)
    results = dict()
    i=0
    for mol in self.getMol():
      i+=1
      if i %20000 == 0:
          print("Processed: %d compounds"%i)
      chembl_id = mol.GetProp('chembl_id') 
      if chembl_id not in ids:
        continue
      results[chembl_id] = AllChem.GetMorganFingerprint(mol, nMorgan).GetNonzeroElements()
    return results

  def getMorganFingerPrintsAll(self, nMorgan):
    results = dict()
    for mol in self.getMol():
      chembl_id = mol.GetProp('chembl_id') 
      results[chembl_id] = AllChem.GetMorganFingerprint(mol, nMorgan).GetNonzeroElements()
    return results
##### end of class SDF ######

def main(argv):
  parser = argparse.ArgumentParser(description='Generate Morgan(3) fingerprints from raw SDF.')
  parser.add_argument('-c', '--compounds', metavar='FILE', help="CSV file of compounds (CHEMBL IDs). If not supplied all compounds are saved.", default=None)
  parser.add_argument('-s', '--sdf', help="Input SDF file.", required=True)
  parser.add_argument('-o', '--out', help="Output file name (CSV file of fingerprints).", required=True)
  parser.add_argument('-r', '--radius', metavar='R', type=int, help="ECFP radius (default 3)", default=3)
  parser.add_argument('--numids', action='store_true')
  args = vars(parser.parse_args())
  return mainf(args["compounds"], args["out"], args["sdf"], nMorgan = args["radius"], numericIds = args["numids"])

def mainf(compoundsFile, outFile, sdfFile, nMorgan = 3, numericIds = False):
  sdf = SDF(sdfFile)
  if compoundsFile is None:
      fp = sdf.getMorganFingerPrintsAll(nMorgan)
  else:
      compoundIDs = getChemblIDs(compoundsFile)
      fp = sdf.getMorganFingerPrints(compoundIDs, nMorgan)
      #fp = {'a':[1,123]}
  saveFingerprints(fp, outFile)

if __name__ == "__main__":
  main(sys.argv[1:])

