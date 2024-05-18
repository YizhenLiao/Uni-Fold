# How to run the inference code
all the inference scripts are in `notebooks` directory.
### denoise mode
* `run_ufconf_denoise.py`: Runs the full backward process, controled by the argument `--from_pdb`.

In default the argument `--from_pdb` is not set, the script will be run with the downloaded dataset **in the volcano engine** for all the PDB files and MSA files in the path provided by argument `--data_path`(default is `/mnt/vepfs/fs_projects/unifold/data_0916/traineval/`) and writes to PDB file. The running parameters (`num_replicas`, `protein ID`...) are defined in the JSON file specified by `-t` argument.  The model (.pt checkpoint) used for inference is specified by `-c`.

If the argument `--from_pdb` is set, the script will be run for a PDB file (will download MSA if needed). The pdb path is specified by the argument `-i` or `--input_pdbs`. The pdb name is specified by the `pdb` param in the JSON input.

To run the script:
```bash
python run_ufconf_denoise.py -t example_diffold/1ake_dataset_monomer.json -c checkpoint.pt -o ./ufconf_out
python run_ufconf_denoise.py -t example_diffold/1ake_from_pdb.json -i input_pdbs/ -c checkpoint.pt -o ./ufconf_out --from_pdb
```

### langevin mode
* `run_ufconf_langevin.py`:  Runs langevin dynamics from a predefined diffusion timestep in the JSON file (MD-like), controled by the argument `--from_pdb`. This is to test the *correlated sampling* of the model besides diffusion dynamics, which is *uncorrelated sampling*.  

To run the script:
```bash
python run_ufconf_langevin.py -t example_diffold/1ake_dataset_monomer.json -c checkpoint.pt -o ./ufconf_out
python run_ufconf_langevin.py -t example_diffold/1ake_from_pdb.json -i input_pdbs/ -c checkpoint.pt -o ./ufconf_out --from_pdb
```

### interpolation mode
* `run_ufconf_interpolate.py`: Runs the interpolation inference between 2 pdbs (from A -> B) where the order is defined in the JSON input file, controled by the argument `--from_pdb`. 

To run the script:
```bash
python run_ufconf_interpolate.py -t example_diffold/2dn1_2dn2_dataset.json -c checkpoint.pt -o ./ufconf_out
python run_ufconf_interpolate.py -t example_diffold/2dn1_2dn2_from_pdb.json -i input_pdbs/ -c checkpoint.pt -o ./ufconf_out --from_pdb
```

# Installation

## Create new conda enviroment
```bash
conda create -n diffold python=3.9
```

## Based on your cuda version, install Pytorch (Using china image if necessary)
```bash
pip3 install torch torchvision torchaudio 
```

## Clone the Uni-core repository and install
```bash
pip install -r requirement.txt
```
If you intend to compile on your slurm machine, connect to a GPU node,cd to Uni-core dir and then
```bash
python setup.py install
```

## Clone diffold repository and install
```bash
pip install .
```

## Install other packages for result postprocessing
```bash
pip install MDAnalysis matplotlib
pip install biopython==1.81
conda install mdtraj -c conda-forge
```

# Input preparation
## task configuration
The json file of task configuration is a dictionary of dictionaries.

The outer dictionary is of items indexed by (unique) task ids.

Available keyword arguments are below.

```python
{
    "residue_idx": "1-140/142-281/283-427/429-573",
    # global index of residues that one wants to sample in ufconf, start from 0, using "/" to separate regions, index is across all chains
    "pdb": "1AEL-12_A",
    # pdb name of one conformation of a protein.
    # a pdb file must exist at path `<input_pdbs>/<pdb>.pdb`.
    "other_pdb": "1URE-8_A", (optional)
    # pdb name of the other conformation of a protein.
    "initial_t": 1.,
    # float in (0, 1], specifies the start of the diffusion process. 
    # with t=1 the pure prior is used; 
    # with t<1, noise is injected (the lower t, the lower noise ratio).
    "num_replica": 10,
    # repeating times of the job (different seeds).
    "inf_steps": 50,
    # number of reverse-diffusion steps.
    # notably, [`initial_t`, 0.] is uniformly splitted into `int_steps`+1 grids.
    "num_steps":10, (optional)
    # number of langevin steps if using langevin sampling mode,
    # number of interpolation steps if using interpolation mode.
    "save_trajectory": True,
    # whether to save the trajectory. Setting `False` is very slightly faster.
}
```

