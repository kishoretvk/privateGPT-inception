#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import sys
import subprocess
import shutil
import tkinter as tk
from tkinter import ttk, filedialog


# Create the chat window
root = tk.Tk()
root.title("PrivateGPT")
root.iconbitmap("PrivateGPT.ico")
root.geometry("400x500")

# Add tabs
notebook = ttk.Notebook(root)

# Create tab frames
chat_tab = tk.Frame(notebook)
newFile_tab = tk.Frame(notebook)
createFile_tab = tk.Frame(notebook)

notebook.add(chat_tab, text="Chat")
notebook.add(newFile_tab, text="Add files")
notebook.add(createFile_tab, text="Create new file")
notebook.pack(expand=True, fill="both")#add the tabs to the window, and allow it to expand with the window.

# Add chat components to the chat tab
chat_box = tk.Text(chat_tab, wrap="word", font=("Arial", 12))
chat_box.pack(side="top", fill="both", expand=True)

input_frame = tk.Frame(chat_tab)
input_field = tk.Entry(input_frame, font=("Arial", 12))
input_field.pack(side="left", fill="both", expand=True, padx=10, pady=10)
input_frame.pack(side="bottom", fill="x")

input_field.bind("<Return>", lambda event: send_message())

# Add components to the add files tab
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        destination_folder = "source_documents/"  # Specify the destination folder

        # Copy the file to the destination folder
        try:
            shutil.copy2(file_path, destination_folder)
            chat_box.insert(tk.END, f"File copied to {destination_folder}\n")
            reload_db()  # Reload the database after copying the file
        except Exception as e:
            chat_box.insert(tk.END, f"Error: {str(e)}\n")

select_file_button = tk.Button(newFile_tab, text="Select File", command=select_file)
select_file_button.pack(padx=20, pady=20)

def run_ingest_script():
    script_path = os.path.join(os.path.dirname(__file__), "ingest.py")
    subprocess.run(["python", script_path])

ingest_button = tk.Button(newFile_tab, text="Run Ingest Script", command=run_ingest_script)
ingest_button.pack(padx=20, pady=10)

ingest_note = tk.Label(newFile_tab, text="After ingesting, relaunch this program.")
ingest_note.pack(padx=20, pady=5)

# Add components to the create file tab
def save_file():
    filename = file_name_entry.get()
    if filename:
        file_path = os.path.join("source_documents", filename + ".txt")
        try:
            with open(file_path, "w") as file:
                file.write(textArea.get("1.0", tk.END))
            chat_box.insert(tk.END, f"File saved: {file_path}\n")
        except Exception as e:
            chat_box.insert(tk.END, f"Error: {str(e)}\n")
    else:
        chat_box.insert(tk.END, "Please enter a file name.\n")

#make the area editable
textArea = tk.Text(createFile_tab, wrap="word", font=("Arial", 12))
textArea.pack(side="top", fill="both", expand=True)
#file-name feild 
file_name_label = tk.Label(createFile_tab, text="File Name:")
file_name_label.pack(padx=20, pady=5)
file_name_entry = tk.Entry(createFile_tab, font=("Arial", 12))
file_name_entry.pack(padx=20, pady=5)

save_button = tk.Button(createFile_tab, text="Save File", command=save_file)
save_button.pack(padx=20, pady=10)


load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    # Create the retriever
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    
    input_field.bind("<Return>", lambda event: send_message(llm=llm,args=args))
   
   # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if(len(query) == 0):
            print("\n\n> Enter a question:")
        if(len(query) > 0):
                    # Get the answer from the chain
            res = qa(query)
            answer, docs = res['result'], [] if args.hide_source else res['source_documents']

            # Print the result
            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)

            # Print the relevant sources used for the answer
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
     
     



def send_message(llm, args):
    queryChat = input_field.get() #get input
    chat_box.insert(tk.END, "\nYou: " + queryChat + "\n") #add user input to window
    chat_box.see(tk.END) #set scroll bar to bottom
    input_field.delete(0, tk.END) #delete input-field contents for next input

    if queryChat == "exit":
        sys.exit()

    elif queryChat is not None: 
        dbChat = Chroma(persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
        retrieverCFhat = dbChat.as_retriever(search_kwargs={"k": target_source_chunks})
        qaChat = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retrieverCFhat)
        # Get the answer from the chain
        resChat = qaChat(queryChat)
        answer, docs = resChat['result'], [] if args.hide_source else resChat['source_documents']
       #resultChat = resChat['result']
        res_stringChat = str(answer) #the output is a dict so this converts it to a string for the chat

    # Print the result
        chat_box.insert(tk.END, "\nNotasloqui: " + res_stringChat + "\n") #add output to window
        chat_box.see(tk.END) #set scroll bar to bottom

def run_ingest_script():
    script_path = os.path.join(os.path.dirname(__file__), "ingest.py")
    subprocess.run(["python", script_path]) 

def run_ingest_script():
    script_path = os.path.join(os.path.dirname(__file__), "ingest.py")
    subprocess.run(["python", script_path])  


def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
