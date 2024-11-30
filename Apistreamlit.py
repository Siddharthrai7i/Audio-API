import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from io import BytesIO
from langchain import HuggingFaceHub
from langchain_community.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain import LLMChain
from scipy.io.wavfile import write
import speech_recognition as sr
import sounddevice as sd
from fpdf import FPDF

# Initialize FastAPI app
app = FastAPI()

# Huggingface API Token
huggingface_api_token = os.getenv("HUGGINGFACEHU_API_TOKEN")

# Clean up text by replacing some special characters
def clean_text(text):
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return text

# Record audio
def recording(duration, samplerate=44100):
    data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait for the recording to finish
    return data

# Convert audio to text
def audiototext(inputfile):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(inputfile) as source:
            audiodata = recognizer.record(source)
            text = recognizer.recognize_google(audiodata)
            return text
    except sr.UnknownValueError:
        return "Sorry, speech was unclear. Please try again."
    except sr.RequestError as e:
        return f"Could not request results from the speech recognition service; {e}"
    except Exception as ex:
        return f"An error occurred: {ex}"

# Use HuggingFace model to summarize text
def predict(text):
    llm = HuggingFaceHub(repo_id="utrobinmv/t5_summary_en_ru_zh_base_2048", model_kwargs={"temperature":0,"max_length":64})
    prompt = PromptTemplate(input_variables=['text'], template='Summarizing the following text in English: {text}')
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(text)
    return summary

# Create PDF from text
def texttopdf(text, output_pdf):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, txt=line)
    pdf.output(output_pdf)

# Define the input model for the API
class AudioRequest(BaseModel):
    duration: int  # in seconds

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    audio_file_path = f"temp_audio_{file.filename}"
    with open(audio_file_path, "wb") as f:
        f.write(await file.read())

    # Convert audio to text
    writtentext = audiototext(audio_file_path)
    summary = clean_text(writtentext)

    # Generate PDF with the summary
    pdf_file = "summary.pdf"
    texttopdf(summary, pdf_file)

    # Read the PDF as a byte stream to send it back to the user
    with open(pdf_file, "rb") as f:
        pdf_content = f.read()

    # Return the PDF as a downloadable file
    return {"summary": summary, "pdf_file": pdf_content}

@app.post("/record_audio/")
async def record_audio(request: AudioRequest):
    # Record audio based on duration
    audio_data = recording(request.duration)

    # Saving the recorded audio as a .wav file
    output_audio_file = "temp_audio.wav"
    write(output_audio_file, 44100, audio_data)

    # Convert audio to text
    writtentext = audiototext(output_audio_file)
    summary = clean_text(writtentext)

    # Generate PDF with the summary
    pdf_file = "summary.pdf"
    texttopdf(summary, pdf_file)

    # Read the PDF as a byte stream to send it back to the user
    with open(pdf_file, "rb") as f:
        pdf_content = f.read()

    return {"summary": summary, "pdf_file": pdf_content}
