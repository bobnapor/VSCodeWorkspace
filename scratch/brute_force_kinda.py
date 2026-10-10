import re
import csv

file_path = 'C:/Users/bobna/Downloads/hand_copied_superflex_rankings.txt'  # Replace with the actual path to your text file

def parse_data(text):
    # Regular expression pattern to extract player information
    pattern = r"\((.*?)\) (.*?), (.*?) \$(\d+) [\d#-]+"

    # Initialize an empty dictionary
    data_dict = {}

    # Find all matches of the pattern in the string
    matches = re.findall(pattern, text)

    # Process each match and populate the dictionary
    for match in matches:
        key = match[1]  # Extract the key from the match
        values = match[3]  # Extract the first two values after the comma
        if values.startswith("0"):
            values = "0"
        if values.endswith("#"):
            values = values[:-1]
        data_dict[key] = values

    return data_dict

player_values = {}

# Open the text file
with open(file_path, 'r') as file:
    # Read the contents line by line
    for line in file:
        player_values.update(parse_data(line))

output_file = 'C:/Users/bobna/Downloads/output.csv'

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    
    for item in player_values.items():
        writer.writerow(item)
        print(item[0] + ' ' + item[1])