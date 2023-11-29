# Usability_Education

# Live Implementation Link
https://appapp-5wipuwsn7kaupp2cajvair.streamlit.app/

## commands
```
python -m pip install -r requirements.txt
streamlit run app.py
```

# Streamlit PDF Analysis Web Application

This Streamlit-based application offers a comprehensive suite of tools for PDF handling and analysis. It integrates various Python libraries to facilitate a wide range of operations on PDF documents, making it a versatile tool for users who need to analyze, modify, or extract information from PDF files.

## Features

### 1. **PDF Report Generation**
   - **Text and Image Extraction**: Extract text and images from PDF files.
   - **Word Cloud Generation**: Create a word cloud from the extracted text to visualize key terms and themes.
   - **Report Creation**: Compile extracted data into a formatted PDF report.

### 2. **Document Viewing**
   - **PDF Viewer**: In-app feature to view the content of uploaded PDF documents.

### 3. **Font Analysis**
   - **Font Extraction**: Analyze and list the fonts used in the PDF, including style and size.

### 4. **Translation**
   - **PDF Translation**: Translate text extracted from PDFs into various languages using Google's translation API.

### 5. **Chat-based PDF Interaction (ChatPDF)**
   - **AI-Enhanced Q&A**: Ask questions about the PDF's content and get answers, leveraging OpenAI's GPT model for intelligent text analysis.

### 6. **Accessibility Features**
   - **Low Vision Support**: Modify PDFs to enhance accessibility for users with low vision, including font adjustments.

### 7. **Sentiment Analysis and Document Classification**
   - **Sentiment Analysis**: Assess the overall sentiment of the text using Expert AI’s sentiment analysis.
   - **Document Classification**: Categorize the document's content into various classes.

### 8. **NLP English Level Analysis**
   - **Linguistic Analysis**: Analyze the text for grammar, parts of speech, entities, relations, and topic analysis.

### 9. **PDF to Audio Conversion**
   - **Text-to-Speech**: Convert PDF text content into audio format, facilitating an auditory mode of consumption.

## Libraries and Technologies Used

- **Streamlit**: For creating the web application.
- **PyPDF2 and Fitz (PyMuPDF)**: For PDF manipulation and rendering.
- **WordCloud**: To generate word clouds from text.
- **OpenAI GPT**: For AI-based text analysis and question-answering.
- **Expert AI Client**: For sentiment analysis, document classification, and NLP.
- **Google Translate API**: For translating text.
- **gTTS (Google Text-to-Speech)**: For converting text to speech.
  
## Who Can Benefit

In essence, this app is a versatile assistant for anyone who needs to work with PDF documents. It streamlines the process of accessing, summarizing, and visualizing PDF content, making it a valuable resource for students, researchers, professionals, and anyone who regularly deals with PDF reports and documents.
