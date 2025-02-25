from pathlib import Path
import subprocess
import mammoth
import os

def convert_folder(input_dir, output_dir):
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # First convert .doc to .docx using LibreOffice
    for doc_file in input_path.glob("*.doc"):
        print(f"🔀 Converting {doc_file.name} to DOCX...")
        subprocess.run([
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
            "--headless",
            "--convert-to", "docx",
            "--outdir", str(input_path),
            str(doc_file)
        ])

    # Now process all .docx files with Mammoth
    for docx_file in input_path.glob("*.docx"):
        try:
            with open(docx_file, "rb") as f:
                result = mammoth.convert_to_markdown(f)
            output_file = output_path / f"{docx_file.stem}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.value)
            print(f"✅ Success: {docx_file.name} → {output_file.name}")
            
        except Exception as e:
            print(f"⚠️ Failed {docx_file.name}: {str(e)}")

# Use local file paths
input_path = "/Users/mpere/Documents/GitHub/Sussex-Strategy-RAG/Original_Documents"
output_path = "/Users/mpere/Documents/GitHub/Sussex-Strategy-RAG/Original_Documents"
convert_folder(input_path, output_path)
