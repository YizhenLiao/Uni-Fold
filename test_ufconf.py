import sys
import subprocess

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_ufconf.py <checkpoint_file>")
        sys.exit(1)

    checkpoint_file = sys.argv[1]

    # Define the paths and other parameters
    script_path = './inference_scripts_ufconf/run_ufconf_denoise.py'
    config_file = 'example_ufconf/1ake_from_pdb.json'
    input_dir = 'example_ufconf/input_pdbs/'
    output_dir = './ufconf_out'
    log_file = './test_log.txt'

    # Build the command to run the Python script
    command = [
        'python', script_path,
        '-t', config_file,
        '-i', input_dir,
        '-c', checkpoint_file,
        '-o', output_dir
    ]

    # Open the log file
    with open(log_file, 'w') as file:
        # Execute the command and redirect stdout and stderr to the log file
        result = subprocess.run(command, stdout=file, stderr=file)

if __name__ == '__main__':
    main()
