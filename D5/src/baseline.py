import sys
import os

def main():
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    summary_length = int(sys.argv[3])
         
    for subdir in os.listdir(input_directory):
        for file in os.listdir(os.path.join(input_directory, subdir)):
            first_file = os.path.join(input_directory, subdir, file)
            # Create the output_directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)
            filename = f"{subdir[:-3]}-A.M.100.{subdir[-3]}.1"
            with open(first_file, 'r') as file, open(os.path.join(output_directory, filename), "w") as output:
                curr_length = 0
                for line in file.readlines():
                    # exclude meta information
                    if not line.startswith("HEADLINE") and not line.startswith("DATE_TIME") \
                        and not line.startswith("DATETIME") and not line.startswith("DATELINE"):
                        sentence = line.strip()
                        if sentence:
                            output.write(sentence + "\n\n")
                            curr_length += len(sentence.split())
                            if curr_length >= summary_length:
                                break
            break # only consider the first file for the baseline

if __name__ == "__main__":   
    main()
        
        