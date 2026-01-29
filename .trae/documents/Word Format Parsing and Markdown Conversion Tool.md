I found the file `大连海事大学硕士学位论文格式模版20181122.doc`.

However, `.doc` is a legacy binary format (Word 97-2003) that is difficult to parse programmatically for **layout/style codes**. Modern tools (like `python-docx` and `pandoc`) require the `.docx` (XML-based) format to accurately read style information (e.g., "Heading 1", "Normal", "Caption").

Here is the plan to solve this:

## 1. Create Style Inspection & Conversion Tool (`tools/docx_inspector.py`)
I will write a Python script that you can use **after saving the file as .docx**. This script will:
- **Parse Layout Codes**: List all *Style Names* defined in the document. This directly answers your request to "parse the layout codes" so you can see exactly what the template uses (e.g., is the title using "Title" or "Heading 1"?).
- **Convert to Markdown**: Generate a Markdown file based on these styles.

## 2. Immediate Action
- **Recommendation**: Please open the `.doc` file in Word/WPS and **"Save As" -> `.docx`**.
- **Alternative**: I will attempt to convert the current `.doc` file using `pandoc` immediately, but please note this often fails or loses style information on binary files without LibreOffice installed.

## 3. Verify
Once you provide the `.docx` version (or if the Pandoc attempt works), we will run the inspector to map the thesis template's specific styles to Markdown headers.
