import tkinter as tk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import json
import os
import docx
import PyPDF2
import requests
import threading
import datetime
import subprocess
import platform

CONFIG_FILE = "server_config.json"

def save_server_config():
    ip = ip_entry.get().strip()
    port = port_entry.get().strip()
    if not ip or not port:
        messagebox.showerror("Missing Info", "Please enter both Server IP and Port.")
        return
    config = {"ip": ip, "port": port}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)
    messagebox.showinfo("Success", "Server configuration saved.")

def load_server_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            ip_entry.delete(0, "end")
            ip_entry.insert(0, config.get("ip", ""))
            port_entry.delete(0, "end")
            port_entry.insert(0, config.get("port", ""))

def select_files():
    filetypes = [("Documents", "*.pdf *.docx *.txt")]
    filenames = filedialog.askopenfilenames(title="Select Sales Reports", filetypes=filetypes)
    if filenames:
        file_listbox.delete(0, "end")
        for file in filenames:
            file_listbox.insert("end", file)
    else:
        messagebox.showinfo("No Selection", "No files selected.")

def preview_selected_file(event):
    try:
        selected_index = file_listbox.curselection()[0]
        file_path = file_listbox.get(selected_index)
        preview_text = ""

        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                preview_text = f.read(1000)
        elif file_path.endswith(".pdf"):
            preview_text = extract_pdf_text_preview(file_path)
        elif file_path.endswith(".docx"):
            preview_text = extract_docx_text_preview(file_path)
        else:
            preview_text = "Preview not supported for this file type."

        preview_box.config(state="normal")
        preview_box.delete("1.0", "end")
        preview_box.insert("end", preview_text)
        preview_box.config(state="disabled")

    except IndexError:
        pass

def extract_pdf_text_preview(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages[:2]:
                text += page.extract_text() or ""
            return text[:1000]
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_docx_text_preview(docx_path):
    try:
        doc = docx.Document(docx_path)
        full_text = []
        for para in doc.paragraphs[:20]:
            full_text.append(para.text)
        return "\n".join(full_text)[:1000]
    except Exception as e:
        return f"Error reading DOCX: {e}"

def upload_files_to_server():
    ip = ip_entry.get().strip()
    port = port_entry.get().strip()
    if not ip or not port:
        messagebox.showerror("Missing Info", "Please enter both Server IP and Port.")
        return

    selected_files = file_listbox.get(0, "end")
    if not selected_files:
        messagebox.showinfo("No Files", "Please select files to upload.")
        return

    url = f"http://{ip}:{port}/upload"

    btn_upload.config(state="disabled")
    progress = tb.Progressbar(app, mode="indeterminate", bootstyle="info")
    progress.pack(pady=10)
    progress.start()

    def upload_thread():
        files_to_send = []
        try:
            for file_path in selected_files:
                file_name = os.path.basename(file_path)
                f = open(file_path, "rb")
                files_to_send.append(('files', (file_name, f)))

            response = requests.post(url, files=files_to_send, timeout=2000)

            for _, file_tuple in files_to_send:
                file_tuple[1].close()

            if response.status_code == 200:
                results = response.json().get("results", [])
                for result in results:
                    filename = result.get("filename", "Unknown File")

                    if "error" in result:
                        app.after(0, lambda f=filename, e=result["error"]:
                                  messagebox.showerror("Summarization Error", f"{f} failed:\n{e}"))
                        continue

                    summary_text = result.get("summary", "")
                    summary_preview_box.config(state="normal")
                    summary_preview_box.delete("1.0", "end")
                    summary_preview_box.insert("end", summary_text)
                    summary_preview_box.config(state="disabled")

                    download_url = f"http://{ip}:{port}{result['summary_pdf']}"
                    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                    base_filename = os.path.splitext(os.path.basename(result['summary_pdf']))[0]
                    default_name = f"{base_filename}_{timestamp}.pdf"

                    save_path = filedialog.asksaveasfilename(
                        defaultextension=".pdf",
                        filetypes=[("PDF Files", "*.pdf")],
                        initialfile=default_name,
                        title=f"Save Summary PDF for {filename}"
                    )

                    if save_path:
                        pdf_response = requests.get(download_url)
                        with open(save_path, "wb") as f:
                            f.write(pdf_response.content)

                        def preview_pdf():
                            try:
                                if platform.system() == "Windows":
                                    os.startfile(save_path)
                                elif platform.system() == "Darwin":
                                    subprocess.call(["open", save_path])
                                else:
                                    subprocess.call(["xdg-open", save_path])
                            except Exception as e:
                                messagebox.showwarning("Preview Failed", f"Could not preview PDF:\n{e}")

                        app.after(0, lambda: (messagebox.showinfo("Summary Saved", f"Saved: {save_path}"), preview_pdf()))

            else:
                app.after(0, lambda: messagebox.showerror(
                    "Upload Failed",
                    f"Server responded with status code {response.status_code}:\n{response.text}"
                ))

        except Exception as e:
            app.after(0, lambda: messagebox.showerror("Error", f"Failed to upload files:\n{e}"))
        finally:
            app.after(0, lambda: (progress.stop(), progress.pack_forget(), btn_upload.config(state="normal")))

    threading.Thread(target=upload_thread, daemon=True).start()

# GUI setup using ttkbootstrap
app = tb.Window(themename="litera")
app.title("🧾 SHARP Sales Report Summarizer")
app.geometry("900x900")

tb.Label(app, text="📊 SHARP AI Summarizer", font=("Segoe UI", 22, "bold"), bootstyle="primary").pack(pady=(15, 5))

ip_frame = tb.Frame(app)
ip_frame.pack(pady=10)
tb.Label(ip_frame, text="Server IP:", font=("Segoe UI", 12)).grid(row=0, column=0, padx=5)
ip_entry = tb.Entry(ip_frame, width=25)
ip_entry.grid(row=0, column=1, padx=5)
tb.Label(ip_frame, text="Port:", font=("Segoe UI", 12)).grid(row=0, column=2, padx=5)
port_entry = tb.Entry(ip_frame, width=10)
port_entry.grid(row=0, column=3, padx=5)

tb.Button(app, text="💾 Save Server Config", bootstyle="success", command=save_server_config).pack(pady=10)
tb.Button(app, text="📂 Select Sales Documents", bootstyle="info", command=select_files).pack(pady=10)
btn_upload = tb.Button(app, text="📤 Upload to Server", bootstyle="primary", command=upload_files_to_server)
btn_upload.pack(pady=10)

file_listbox = tk.Listbox(app, height=8, font=("Segoe UI", 10))
file_listbox.pack(padx=10, pady=5)
file_listbox.bind("<<ListboxSelect>>", preview_selected_file)

tb.Label(app, text="📄 Original File Preview", font=("Segoe UI", 12, "bold")).pack()
preview_box = tb.Text(app, height=10, wrap="word", state="disabled", font=("Segoe UI", 10))
preview_box.pack(padx=10, pady=5)

tb.Label(app, text="🧠 AI Summary Preview", font=("Segoe UI", 12, "bold")).pack()
summary_preview_box = tb.Text(app, height=12, wrap="word", state="disabled", font=("Segoe UI", 10))
summary_preview_box.pack(padx=10, pady=10)

load_server_config()
app.mainloop()
