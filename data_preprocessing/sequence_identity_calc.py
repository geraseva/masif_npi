from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import pairwise2
import sys

from tqdm import tqdm

pdb_dir = '/home/domain/data/geraseva/masif/data/masif_npi/data_preparation/00-raw_pdbs'


def extract_sequences(infile: str ):
    
    seqs=[]
    with open(infile,'r') as f:
        for pdb_id in tqdm(f.readlines()):
            chains=pdb_id.split('_')[1]
            pdb_id=pdb_id.split('_')[0]
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure(f"{pdb_id}_{chains}",f"{pdb_dir}/{pdb_id}_{chains}.pdb")[0]
            for chain in chains:
                seq=Seq('')
                ppb=PPBuilder()
                try:
                    for pp in ppb.build_peptides(struct[chain]):
                        seq=seq+pp.get_sequence()
                except:
                    continue
                seqs.append(SeqRecord(seq, id=f"{pdb_id}_{chain}"))

    SeqIO.write(seqs, infile.split('.')[0]+'.fasta', "fasta")  
    return 0


def calc_seq_id(fa1, fa2, outfile):
    with open(outfile, 'a') as of:
        records1=list(SeqIO.parse(fa1, "fasta"))
        for r1 in tqdm(records1):
            records2=SeqIO.parse(fa2, "fasta")
            for r2 in records2:
                al = pairwise2.align.globalxx(r1.seq, r2.seq)
                try:
                    identity=al[0].score/min(len(r1.seq),len(r2.seq))
                except:
                    identity=0
                of.write(f"{r1.id} {r2.id} {identity}\n")
    return 0
 
for infile in sys.argv[1:]:
    #extract_sequences(infile)
    calc_seq_id(infile.split('.')[0]+'.fasta', infile.split('.')[0].replace('testing','training')+'.fasta', infile.split('.')[0]+'_identities.list')