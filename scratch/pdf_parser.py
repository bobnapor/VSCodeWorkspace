import PyPDF2
import re

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text

#last column is getting thrown out
def parse_data(text):
    # Regular expression pattern to extract player information
    pattern = r"\((.*?)\) (.*?), (.*?) \$([\d#]+) \d+"

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

# Provide the path to your PDF file
pdf_file_path = 'C:/Users/bobna/Downloads/NFL23_CS_Super.pdf'

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_file_path)

# Parse the extracted text into a dataset
parsed_dataset = parse_data(pdf_text)

# Print the parsed dataset
for data in parsed_dataset:
    print(data)
