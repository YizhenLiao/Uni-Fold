# Notebook tutorials
## denoise mode
* `run_ufconf_denoise.py`: Runs the full backward process, controled by the argument `--from_pdb`.

In default the argument `--from_pdb` is not set, the script will be run with the downloaded dataset **in the volcano engine** for all the PDB files and MSA files in the path provided by argument `--data_path`(default is `/mnt/vepfs/fs_projects/unifold/data_0916/traineval/`) and writes to PDB file. Number of trajectories can be controlled from a JSON input with `num_replicas` param. The pdb id is specified by the `id` param in the JSON input

If the argument `--from_pdb` is set, the script will be run for a PDB file (will download MSA if needed). The pdb path is specified by the argument `-i` or `--input_pdbs`. The pdb name is specified by the `pdb` param in the JSON input.

To run the script:
```bash
python run_ufconf_denoise.py -t example_diffold/1ake_dataset_monomer.json -c checkpoint.pt -o ./ufconf_out
python run_ufconf_denoise.py -t example_diffold/1ake_from_pdb.json -i input_pdbs/ -c checkpoint.pt -o ./ufconf_out --from_pdb
```

## langevin mode
* `run_ufconf_langevin.py`:  Runs langevin dynamics from a predefined diffusion timestep in the JSON file (MD-like), controled by the argument `--from_pdb`. This is to test the *correlated sampling* of the model besides diffusion dynamics, which is *uncorrelated sampling*.  

To run the script:
```bash
python run_ufconf_langevin.py -t example_diffold/1ake_dataset_monomer.json -c checkpoint.pt -o ./ufconf_out
python run_ufconf_langevin.py -t example_diffold/1ake_from_pdb.json -i input_pdbs/ -c checkpoint.pt -o ./ufconf_out --from_pdb
```

## interpolation mode
* `run_ufconf_interpolate.py`: Runs the interpolation inference between 2 pdbs (from A -> B) where the order is defined in the JSON input file, controled by the argument `--from_pdb`. 

To run the script:
```bash
python run_ufconf_interpolate.py -t example_diffold/2dn1_2dn2_dataset.json -c checkpoint.pt -o ./ufconf_out
python run_ufconf_interpolate.py -t example_diffold/2dn1_2dn2_from_pdb.json -i input_pdbs/ -c checkpoint.pt -o ./ufconf_out --from_pdb
```