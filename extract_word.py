import PyPDF2

# Open the PDF file
pdf_file = open('codefest.pdf', 'rb')

# Create a PDF reader object
pdf_reader = PyPDF2.PdfReader(pdf_file)

# Loop through each page of the PDF
for page_num in range(len(pdf_reader.pages)):
    
    # Get the text from the current page
    page = pdf_reader.pages[page_num]
    text = page.extract_text()
    
    # Split the text into individual words
    words = text.split()

    print(words)
    
    # Define categories and keywords to look for
    categories = ['fruit', 'vegetable', 'grain']
    keywords = {'fruit': ['Python', 'orange', 'banana'],
                'vegetable': ['carrot', 'celery', 'spinach'],
                'grain': ['rice', 'wheat', 'oats']}
    
    # Initialize empty dictionary to store category counts
    category_counts = {category: 0 for category in categories}
    
    # Loop through each word and check if it matches a keyword
    for word in words:
        for category in categories:
            if word.lower() in keywords[category]:
                category_counts[category] += 1
                
    # Print the category counts for the current page
    print(f'Page {page_num + 1}: {category_counts}')
    
# Close the PDF file
pdf_file.close()
