import PyPDF2

pdf_file = open('scdc-resume.pdf', 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)

for page_num in range(len(pdf_reader.pages)):
    
    page = pdf_reader.pages[page_num]
    text = page.extract_text()
    words = text.split()

    print(words)

    categories = ['Skills', 'Education', 'Soft_Skills']
    keywords = {'Skills': ['Python', 'Java', 'R'],
                'Education': ['B.tech', 'Master', 'Phd'],
                'Soft_Skills': ['Leadership', 'Teamplayer', 'Motivated']}
    
    category_counts = {category: 0 for category in categories}
    
    for word in words:
        for category in categories:
            if word.lower() in keywords[category]:
                category_counts[category] += 1
                
    print(f'Page {page_num + 1}: {category_counts}')
    
pdf_file.close()
