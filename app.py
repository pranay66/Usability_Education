import streamlit as st # web 
#pdf handling
import PyPDF2
import fitz
import base64

# os module 
import os
import io

from wordcloud import WordCloud

# plots and computer vision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from pdf2image import convert_from_bytes
import cv2
from skimage import io
from PIL import Image 
# import aspose.words as aw
# import ironpdf

# from md2pdf.core import md2pdf

def pdfGen(pages, lines, images, word, wordcloudpath, summary, pdftable, filepath):
    # Read the content template from a Markdown file
    with open("format.md", "r") as md_file:
        text_template = md_file.read()

    # Format the text with the provided data
    text = text_template.format(
        pages, lines, images, word, wordcloudpath, 
         summary
    )

    # Create a new PDF document
    doc = fitz.open()

    # Add a new page to the document(
    page = doc.new_page()

    # Add the text data to the page
    page.insert_text((100, 100), text, fontsize=12)

    # Save the PDF document
    doc.save(filepath)
        


import openai
st.set_option('deprecation.showPyplotGlobalUse', False)
from streamlit_javascript import st_javascript


def analyze_color_distribution(image_path):
    image = cv2.imread(image_path)
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

    return hist

# make tmp dir
os.makedirs("tmp", exist_ok=True)

# Define the main title for the app
st.title("PDF Evalutaion [Team Usability]")

# Create a sidebar menu for navigation
menu = st.sidebar.selectbox("Navigation", ["Guide", "Generate Report","Document Viewing","Font Analysis"])

# Page 1: Guide
if menu == "Guide":
    st.write("## Welcome to the PDF Report Generator and Analyzer.")
    st.write(open("description.txt").read())
    st.write("Please navigate to other pages for specific tasks.")

@st.cache_data
def readFile(uploaded_file):
    pdf_file = PyPDF2.PdfReader(uploaded_file)
    pdf_text=""
    # progress_text = "Reading Pdf Text... In Progress "
    # my_bar = st.progress(0.0, text=progress_text)
    try:
        for page_num,page in  enumerate(pdf_file.pages):
                pdf_text+="\n"
                pdf_text += page.extract_text()
                percent = (page_num )/pagesCount
                # my_bar.progress(percent,text = progress_text+" "+str(percent*100) )
        # my_bar.empty()
    except:
        pass
    return pdf_text

@st.cache_data
def imagesExtract(uploaded_file):
    count = 0
    img_data = []
    pdf_file = PyPDF2.PdfReader(uploaded_file)
    try:
        for page_num,page in  enumerate(pdf_file.pages):
            for image_file_object in page.images:
                fileName = str(count) + image_file_object.name
                image_file_name = os.path.join(folderName,fileName)
                with open(image_file_name, "wb") as fp:
                    fp.write(image_file_object.data)
                    count += 1
                # st.image(image_file_name)
                img_data.append(fileName)
    except:
        pass
    return img_data

@st.cache_data
def generate_word_cloud(text,filename):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    st.write("## word cloud")
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig("{}.png".format(filename), format="png")
    plt.show()
    st.image("{}.png".format(filename))
    return "{}.png".format(filename)
    

@st.cache_data
def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    image_paths = []
    for page_num in range(len(doc)):
        # st.write(page_num)
        page = doc.load_page(page_num)
        image = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        # image_bytes = image.getBits()
        # image_data = np.frombuffer(image_bytes, dtype=np.uint8)
        # image_data = image_data.reshape(image.height, image.width, 3)
        # image_pil = Image.fromarray(image_data)
        image_path = f"page_{page_num + 1}.png"
        # image_pil.save(image_path)
        # st.write(dir(image))
        image.save(str(image_path))
        image_paths.append(image_path)
    return image_paths

@st.cache_data
def extract_colors_from_pdf(pdf_path):
    monochrome_pages = 0
    unicolor_pages = 0
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        img = page.get_pixmap()
        
        if img.is_monochrome:
            monochrome_pages += 1
        else:
            unicolor_pages += 1
    
    return monochrome_pages, unicolor_pages


def plot_histogram(monochrome_pages, unicolor_pages):
    labels = ['Monochrome', 'Unicolor']
    page_counts = [monochrome_pages, unicolor_pages]
    
    plt.bar(labels, page_counts)
    plt.xlabel('Page Type')
    plt.ylabel('Page Count')
    # plt.title('Monochrome vs. Unicolor Pages')
    # plt.show()
    return plt

@st.cache_data
def openaisummarize(pdftext,wordcount):
    openai.api_key = st.secrets['API_Key']
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": f"summarize following content for me in {wordcount} words:\n"+pdftext[:4000]}],
    temperature=0,
    max_tokens=1024
    )
    return response["choices"][0]["message"]["content"]
    # st.write(response["choices"][0]["message"]["content"])



# Page 2: Generate Report
if menu == "Generate Report":
    st.header("Generate Report from PDF")

    # Allow the user to upload a PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # create sub floder in tmp on uploaded file name without extension
        filename, _ = os.path.splitext(uploaded_file.name)
        folderName = f"./tmp/{filename}"
        try:
            os.mkdir(folderName)
        except FileExistsError:
            pass
        st.write("PDF file uploaded successfully!")

        st.subheader("Report from PDF:")
        
        # Extract text from the uploaded PDF
        pdf_text = ""
        pdf_file = PyPDF2.PdfReader(uploaded_file)

        # st.write(dir(pdf_file))
        pagesCount = (len(pdf_file.pages))
        
        page = None

        pdf_text=readFile(uploaded_file)

        st.write(pdf_file.metadata)
        
        with st.expander("Extracted Text"):
            st.write(pdf_text)

        summary = None
        with st.expander("Summary"):
            # select number of words
            numWords = st.number_input('Number of Words', min_value=10, max_value=200)
            summary =  openaisummarize(pdf_text,30)
            st.write(openaisummarize(pdf_text,numWords))
        
        # images
        img_data = imagesExtract(uploaded_file)
        # st.write(dir(page))


        
        with st.expander("Extracted Images"):
            if len(img_data)>0:
                options = st.multiselect("Select images to dispaly",img_data)
                # if(len(options)>3):
                cols = st.columns(3)
                for count,option in enumerate(options):
                    cols[count%3].image(os.path.join(folderName,option))
            st.write(img_data)

        
        st.write("## Number of pages:",pagesCount)
        lines = pdf_text.split("\n")
        st.write("## line count :",len(lines))
        words = []
        for i in lines:
            words.extend(i.split(" "))
        st.write("## Word count :",len(words))
        st.write("## Images count :",len(img_data))

        cloudpath = ""
        if(len(words)): cloudpath=generate_word_cloud(pdf_text,filename)
        else: st.write("Cannot generate Word cloud, No text found in Pdf")

        # st.write("## word cloud")
        # pdfGen(pages,lines,images,word,wordcloudpath,summary,pdftable,fielpath)
        pdfGen(
            pagesCount,len(lines),len(words),len(img_data),"![]({})".format(cloudpath),summary,"",f"{filename}.pdf"
        )

        with open(f"{filename}.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        st.download_button(label="Export_Report",
                            data=PDFbyte,
                            file_name="report.pdf",
                            mime='application/octet-stream')


        # st.image("{}.png".format(filename))
        with open(os.path.join(uploaded_file.name),"wb") as f: 
                f.write(uploaded_file.getbuffer()) 


        st.pyplot(plot_histogram(*extract_colors_from_pdf(uploaded_file.name)))

        # image analysis

        if uploaded_file:
            st.markdown("### Page Selection")
            # st.write(dir(uploaded_file))
            

            selected_page = st.selectbox("Select a page to analyze", list(range(1, len(fitz.open(uploaded_file.name)) + 1)))
            a = st.empty()

            if st.button("Analyze"):
                a.empty()
                a.cols= a.columns(2)
                st.markdown(f"Analyzing page {selected_page}")
                image_paths = pdf_to_images(uploaded_file.name)
                # a.write(image_paths)
                a.cols[0].image(image_paths[selected_page - 1], use_column_width=True,caption=f"Page {i}")
                import cv2
                import matplotlib.pyplot as plt
                image = cv2.imread(image_paths[selected_page - 1])
                for i, col in enumerate(['b', 'g', 'r']):
                    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                    # 
                    plt.plot(hist, color = col)
                    plt.ylim([0.0,0.4*1e6])
                    plt.xlim([0, 256])
                    
                a.cols[1].pyplot(plt)

        

def displayPDF(upl_file, width):
    # Read file as bytes:
    bytes_data = upl_file.getvalue()
    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")
    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width={str(width)} height={str(width*4/3)} type="application/pdf"></iframe>'
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


if menu == "Document Viewing":
    st.header("View Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    if uploaded_file is not None:
        ui_width = st_javascript("window.innerWidth")
        displayPDF(uploaded_file, ui_width -2)


if menu == "Font Analysis":
    st.header("Font Analysis Page")
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

    if uploaded_file is not None:
        filename, _ = os.path.splitext(uploaded_file.name)
        folderName = f"./tmp/{filename}"
        try:
            os.mkdir(folderName)
        except FileExistsError:
            pass

        # Save the uploaded PDF file to a temporary location
        with open(os.path.join(folderName, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("PDF file uploaded successfully!")

        st.subheader("Font Analysis:")

        pdf_file_path = os.path.join(folderName, uploaded_file.name)

        doc = fitz.open(pdf_file_path)
        font_info = {}

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # Load the page
            fonts = page.get_fonts()
            page_str = str(page_num+1)
            st.write("Page  " + page_str + " font analysis")
            data_frame = pd.DataFrame(fonts,columns=["Font Size","Font type","Font Sub Type","Font Name","Font Descriptor","Font encoding type"])
            # print(type(data_frame))
            st.dataframe(data_frame)
        doc.close()
