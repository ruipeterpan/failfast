# %%
import os
import html
import json
import math
import pickle
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from playwright.sync_api import sync_playwright

# colors = [  # Gemini 3's defaults
#     "rgba(255, 173, 173, 0.5)", 
#     "rgba(255, 214, 165, 0.5)", 
#     "rgba(253, 255, 182, 0.5)", 
#     "rgba(202, 255, 191, 0.5)", 
#     "rgba(155, 246, 255, 0.5)", 
#     "rgba(160, 196, 255, 0.5)", 
#     "rgba(189, 178, 255, 0.5)",
#     "rgba(255, 198, 255, 0.5)"
# ]

colors = [
    # "rgba(116, 198, 157, 1)",
    # "rgba(149, 213, 178, 1)",
    "rgba(183, 228, 199, 1)",
    # "rgba(200, 236, 210, 1)",
    "rgba(216, 243, 220, 1)",
]

def save_sd_trajectory_html(pickled_data, tokenizer, filename="trajectory.html", font_family="monospace"):
    """
    Generates an HTML file visualizing the speculative decoding trajectory.
    Enforces a fixed width to ensure PDF generation matches screen layout exactly.
    """

    # --- 2. HTML Header & CSS ---
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Speculative Decoding Trajectory</title>
        <style>
            :root {{
                /* --- CONFIGURATION --- */
                --font-family: {font_family};
                --font-size: 8px;              
                --line-height-mult: 1.8;
                --rejected-opacity: 0.45;      
                --border-width: 1.5px;         
                --border-color: #333;
                --container-border: 1px solid #ccc;
                --strikethrough-width: 1px; 
                
                /* FIX: Define a strict fixed width for consistency */
                --container-width: 900px;
            }}
            
            * {{
                box-sizing: border-box;
            }}

            /* --- PRINT SETTINGS --- */
            @media print {{
                @page {{
                    margin: 0;
                    size: auto; 
                }}
                body {{
                    -webkit-print-color-adjust: exact !important;
                    print-color-adjust: exact !important;
                    background-color: white !important;
                    margin: 1cm !important;
                }}
                .container {{
                    /* FIX: Do NOT expand to 100%. Keep the strict width. */
                    width: var(--container-width) !important;
                    max-width: var(--container-width) !important;
                    
                    border: var(--container-border) !important;
                    box-shadow: none !important;
                    margin: 0 !important;
                }}
            }}

            /* --- SCREEN SETTINGS --- */
            body {{
                font-family: var(--font-family);
                padding: 40px;
                background-color: #f9f9f9;
                color: #333;
            }}

            .container {{
                background: white;
                padding: 15px;
                border: var(--container-border);
                display: flex;
                flex-direction: row;
                
                /* FIX: Use fixed width instead of max-width to prevent reflows */
                width: var(--container-width);
                margin: 0 auto;
                
                align-items: flex-start;
            }}

            .line-numbers {{
                text-align: right;
                padding-right: 10px;
                border-right: 1px solid #eee;
                margin-right: 10px;
                color: #bbb;
                user-select: none;
                
                font-size: var(--font-size);
                font-family: var(--font-family);
                line-height: var(--line-height-mult);
                
                margin-top: 0;
                padding-top: 0;
                min-width: 25px; 
            }}

            .content {{
                white-space: pre-wrap; 
                word-break: break-word;
                width: 100%;
                
                font-size: var(--font-size);
                font-family: var(--font-family);
                line-height: var(--line-height-mult);
                
                margin-top: 0;
                padding-top: 0;
            }}

            .token {{
                display: inline; 
                padding: 1px 0; 
                -webkit-box-decoration-break: clone;
                box-decoration-break: clone;
                border: var(--border-width) solid transparent;
                border-radius: 0;
            }}

            .rejected {{
                text-decoration: none; 
                background-image: linear-gradient(
                    to bottom, 
                    transparent calc(50% - var(--strikethrough-width)/2), 
                    #333 calc(50% - var(--strikethrough-width)/2), 
                    #333 calc(50% + var(--strikethrough-width)/2), 
                    transparent calc(50% + var(--strikethrough-width)/2)
                );
                background-origin: border-box; 
                background-clip: border-box;
                background-repeat: no-repeat;
                background-size: 100% 100%;
                opacity: var(--rejected-opacity);
            }}

            .corrected {{
                font-weight: normal; 
                border-color: var(--border-color); 
            }}
            
            .token:hover {{
                cursor: default;
                position: relative;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="line-numbers" id="line-numbers">1</div>
            <div class="content" id="text-content">
    """

    # --- 3. Process Logic ---
    stats_each_round = pickled_data.get("stats_each_round", [])
    
    color_idx = 0 
    
    def create_span(token_ids, css_class=""):
        nonlocal color_idx
        spans = ""
        
        for tid in token_ids:
            text = tokenizer.decode([tid], skip_special_tokens=False)
            if text == "\n":
                text = "\n "
            
            safe_text = html.escape(text)
            bg_color = colors[color_idx % len(colors)]
            
            spans += (f'<span class="token {css_class}" '
                      f'style="background-color: {bg_color};" '
                      f'title="ID: {tid}">{safe_text}</span>')
            
            color_idx += 1
        
            print(f"Text {text} rendered with class '{css_class}'")
        
        return spans

    for round_id, round_data in enumerate(stats_each_round):
        draft_proposal = round_data["~draft_proposal"]
        accepted_len = round_data["accepted_len"]
        proposal_len = len(draft_proposal)

        draft_accepted = draft_proposal[:accepted_len]
        draft_rejected = draft_proposal[accepted_len:]
        
        target_token_id = None
        if accepted_len < proposal_len:
            target_token_id = round_data["final_token"]
        elif accepted_len == proposal_len:
            target_token_id = round_data["bonus_token"]

        html_content += create_span(draft_accepted)
        if len(draft_rejected) > 0:
            html_content += create_span(draft_rejected, css_class="rejected")
        if target_token_id is not None:
            html_content += create_span([target_token_id], css_class="corrected")

    # --- 4. Finalize HTML ---
    html_content += """
            </div>
        </div>
        
        <script>
            function updateLineNumbers() {
                const content = document.getElementById('text-content');
                const lineNumbers = document.getElementById('line-numbers');
                const style = window.getComputedStyle(content);
                const lineHeight = parseFloat(style.lineHeight);
                const height = content.getBoundingClientRect().height;
                
                if (lineHeight > 0) {
                    const lines = Math.round(height / lineHeight);
                    let numHtml = '';
                    for(let i=1; i<=lines; i++) {
                        numHtml += i + '<br>';
                    }
                    lineNumbers.innerHTML = numHtml;
                }
            }
            window.addEventListener('load', updateLineNumbers);
            window.addEventListener('resize', updateLineNumbers);
            setTimeout(updateLineNumbers, 300);
        </script>
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Visualization saved to: {os.path.abspath(filename)}")



def save_sd_accepted_trajectory_html(pickled_data, tokenizer, filename="accepted_trajectory.html", font_family="monospace"):
    """
    Generates an HTML file visualizing ONLY the accepted tokens.
    Enforces a fixed width (900px) to ensure PDF generation matches screen layout exactly.
    """
    
    # --- 1. Define Color Palette ---

    # --- 2. HTML Header & CSS ---
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Accepted Trajectory</title>
        <style>
            :root {{
                /* --- CONFIGURATION --- */
                --font-family: {font_family};
                --font-size: 8px;              
                --line-height-mult: 1.8;       
                --border-width: 1.5px;         
                --container-border: 1px solid #ccc;
                
                /* FIX: Define a strict fixed width for consistency */
                --container-width: 900px;
            }}
            
            * {{
                box-sizing: border-box;
            }}

            /* --- PRINT SETTINGS --- */
            @media print {{
                @page {{
                    margin: 0;
                    size: auto; 
                }}
                body {{
                    -webkit-print-color-adjust: exact !important;
                    print-color-adjust: exact !important;
                    background-color: white !important;
                    margin: 1cm !important;
                }}
                .container {{
                    /* FIX: Force exact width in print mode, do not expand to 100% */
                    width: var(--container-width) !important;
                    max-width: var(--container-width) !important;
                    
                    border: var(--container-border) !important;
                    box-shadow: none !important;
                    margin: 0 !important;
                }}
            }}

            /* --- SCREEN SETTINGS --- */
            body {{
                font-family: var(--font-family);
                padding: 40px;
                background-color: #f9f9f9;
                color: #333;
            }}

            .container {{
                background: white;
                padding: 15px;
                border: var(--container-border);
                display: flex;
                flex-direction: row;
                
                /* FIX: Use fixed width instead of max-width */
                width: var(--container-width);
                margin: 0 auto;
                
                align-items: flex-start;
            }}

            .line-numbers {{
                text-align: right;
                padding-right: 10px;
                border-right: 1px solid #eee;
                margin-right: 10px;
                color: #bbb;
                user-select: none;
                
                font-size: var(--font-size);
                font-family: var(--font-family);
                line-height: var(--line-height-mult);
                
                margin-top: 0;
                padding-top: 0;
                min-width: 25px; 
            }}

            .content {{
                white-space: pre-wrap; 
                word-break: break-word;
                width: 100%;
                
                font-size: var(--font-size);
                font-family: var(--font-family);
                line-height: var(--line-height-mult);
                
                margin-top: 0;
                padding-top: 0;
            }}

            .token {{
                display: inline; 
                padding: 1px 0; 
                
                -webkit-box-decoration-break: clone;
                box-decoration-break: clone;
                
                /* Transparent border ensures layout consistency with the full trajectory plot */
                border: var(--border-width) solid transparent;
                border-radius: 0;
            }}
            
            .token:hover {{
                cursor: default;
                position: relative;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="line-numbers" id="line-numbers">1</div>
            <div class="content" id="text-content">
    """

    # --- 3. Process Logic ---
    stats_each_round = pickled_data.get("stats_each_round", [])
    
    color_idx = 0 
    
    def create_span(token_ids, css_class=""):
        nonlocal color_idx
        spans = ""
        
        for tid in token_ids:
            text = tokenizer.decode([tid], skip_special_tokens=False)
            
            # Pad newline characters
            if text == "\n":
                text = "\n "
            
            safe_text = html.escape(text)
            bg_color = colors[color_idx % len(colors)]
            
            spans += (f'<span class="token {css_class}" '
                      f'style="background-color: {bg_color};" '
                      f'title="ID: {tid}">{safe_text}</span>')
            
            color_idx += 1
        return spans

    # 3b. Add Generated Trajectory
    for round_id, round_data in enumerate(stats_each_round):
        draft_proposal = round_data["~draft_proposal"]
        accepted_len = round_data["accepted_len"]
        proposal_len = len(draft_proposal)

        # 1. Identify Tokens
        draft_accepted = draft_proposal[:accepted_len]
        
        target_token_id = None
        if accepted_len < proposal_len:
            target_token_id = round_data["final_token"]
        elif accepted_len == proposal_len:
            target_token_id = round_data["bonus_token"]

        # 2. Render Tokens (All with default transparency)
        
        # A) Draft Accepted
        html_content += create_span(draft_accepted)
        
        # B) Target Token (No special CSS class, just background)
        if target_token_id is not None:
            html_content += create_span([target_token_id])

    # --- 4. Finalize HTML ---
    html_content += """
            </div>
        </div>
        
        <script>
            function updateLineNumbers() {
                const content = document.getElementById('text-content');
                const lineNumbers = document.getElementById('line-numbers');
                
                const style = window.getComputedStyle(content);
                const lineHeight = parseFloat(style.lineHeight);
                const height = content.getBoundingClientRect().height;
                
                if (lineHeight > 0) {
                    const lines = Math.round(height / lineHeight);
                    
                    let numHtml = '';
                    for(let i=1; i<=lines; i++) {
                        numHtml += i + '<br>';
                    }
                    lineNumbers.innerHTML = numHtml;
                }
            }
            
            window.addEventListener('load', updateLineNumbers);
            window.addEventListener('resize', updateLineNumbers);
            setTimeout(updateLineNumbers, 300);
        </script>
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Accepted token visualization saved to: {os.path.abspath(filename)}")

def convert_html_to_cropped_pdf(html_filepath, pdf_filepath):
    """
    Opens the HTML file, removes all margins via CSS injection,
    forces background graphics, and saves a PDF cropped exactly to the .container.
    """
    abs_html_path = f"file://{os.path.abspath(html_filepath)}"
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # FIX: Set a large viewport to ensure the 900px container fits comfortably
        page.set_viewport_size({"width": 1200, "height": 800})
        
        page.goto(abs_html_path)
        
        # Inject CSS to force layout and FORCE GRAPHICS
        page.add_style_tag(content="""
            @media print {
                @page { margin: 0; size: auto; }
                
                body { 
                    margin: 0 !important; 
                    padding: 0 !important; 
                }
                
                .container { 
                    margin: 0 !important; 
                    position: absolute !important; 
                    top: 0 !important; 
                    left: 0 !important;
                    box-shadow: none !important;
                }
                
                * {
                    -webkit-print-color-adjust: exact !important;
                    print-color-adjust: exact !important;
                }
            }
        """)

        page.emulate_media(media="print")
        
        page.wait_for_selector(".container")
        bbox = page.locator(".container").bounding_box()
        
        if bbox:
            page.pdf(
                path=pdf_filepath,
                width=f"{bbox['width'] + 1}px", 
                height=f"{bbox['height'] + 1}px",
                print_background=True,
                margin={'top': '0px', 'right': '0px', 'bottom': '0px', 'left': '0px'}
            )
            print(f"Perfectly cropped PDF saved to: {pdf_filepath}")
        else:
            print("Error: Could not find .container element.")
            
        browser.close()



# %%
data_dir = "/data2/USERNAME/failfast" 
# pickle_filename = "pickles/Qwen2.5-32B-Instruct/math/2/dllm_0.05_df_0.4_60_10/1024.pickle"
pickle_filename = "pickles/Qwen2.5-32B-Instruct/math/2/ar_None_sf_8/1024.pickle"
pickle_fullpath = os.path.join(data_dir, pickle_filename)




html_pdf_dir = pickle_fullpath.replace("pickles", "html_pdf")
if not os.path.exists(os.path.dirname(html_pdf_dir)):
    os.makedirs(os.path.dirname(html_pdf_dir), exist_ok=True)
    
with open(pickle_fullpath, "rb") as f:
    pickled_data = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
save_sd_trajectory_html(pickled_data, tokenizer, filename=html_pdf_dir.replace(".pickle", "_trajectory.html"))
save_sd_accepted_trajectory_html(pickled_data, tokenizer, filename=html_pdf_dir.replace(".pickle", "_accepted_trajectory.html"))

# Then convert it
convert_html_to_cropped_pdf(html_pdf_dir.replace(".pickle", "_trajectory.html"), html_pdf_dir.replace(".pickle", "_trajectory_figure.pdf"))
convert_html_to_cropped_pdf(html_pdf_dir.replace(".pickle", "_accepted_trajectory.html"), html_pdf_dir.replace(".pickle", "_accepted_trajectory_figure.pdf"))
# %%
