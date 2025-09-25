# app.py (Flask Backend - Improved LLM Response Handling)

import os
import uuid 
import re 
import json # Ensure json is imported
from flask import Flask, request, jsonify
from flask_cors import CORS 
import pandas as pd
import google.generativeai as genai 

app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'json'} 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

GEMINI_API_KEY = None
try:
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API Key configured.")
    else:
        print("WARNING: GEMINI_API_KEY not found.")
except ImportError:
    print("WARNING: python-dotenv not installed.")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API Key configured from environment.")
    else:
         print("WARNING: GEMINI_API_KEY not found.")

uploaded_file_details = {
    "filename": None, "filepath": None,
    "dataframe": None, "columns": []
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file_route():
    global uploaded_file_details
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        original_filename = file.filename
        extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(filepath)
            if extension == 'csv': df = pd.read_csv(filepath, low_memory=False) 
            elif extension == 'json':
                try: df = pd.read_json(filepath, orient='records')
                except ValueError:
                    try: df = pd.read_json(filepath)
                    except Exception as json_e: raise ValueError(f"Could not parse JSON. Error: {json_e}")
            else: raise ValueError("Unsupported file type.")

            if uploaded_file_details["filepath"] and os.path.exists(uploaded_file_details["filepath"]):
                os.remove(uploaded_file_details["filepath"])

            uploaded_file_details["filename"] = original_filename
            uploaded_file_details["filepath"] = filepath
            uploaded_file_details["dataframe"] = df
            uploaded_file_details["columns"] = df.columns.tolist()
            
            df_head_safe = df.head().where(pd.notnull(df.head()), None)
            sample_data = df_head_safe.to_dict(orient='records')

            return jsonify({
                "message": f"File '{original_filename}' uploaded.",
                "filename": original_filename, "columns": df.columns.tolist(),
                "rowCount": len(df), "sampleData": sample_data
            }), 200
        except Exception as e:
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({"error": f"Failed to process file: {str(e)}"}), 500
    else:
        return jsonify({"error": "File type not allowed (CSV/JSON only)."}), 400

@app.route('/api/query', methods=['POST'])
def query_data():
    global uploaded_file_details
    if uploaded_file_details["dataframe"] is None: return jsonify({"error": "No data uploaded."}), 400
    user_query = request.json.get('query')
    if not user_query: return jsonify({"error": "No query provided"}), 400
    if not GEMINI_API_KEY: return jsonify({"error": "Gemini API Key not configured."}), 500

    df = uploaded_file_details["dataframe"]
    column_details = "\n".join([f"- '{c}' (type: {df[c].dtype}, examples: {df[c].dropna().unique()[:3].tolist()})" for c in df.columns])
    
    prompt = f"""
        You are an AI assistant for a Business Intelligence dashboard.
        The dataset columns are:
        {column_details}

        User question: "{user_query}"

        Provide a JSON response. 
        If the question is directly answerable by filtering, aggregating, or describing the data, 
        use this structure (fill ALL keys, using null/[] if not applicable):
        {{
          "filters": [
            {{"column": "ColumnName", "operator": "equals/contains/greater_than/less_than", "value": "FilterValue"}}
          ],
          "analysis_type": "aggregation/filter_and_display/trend/describe",
          "aggregation": {{"column_to_aggregate": "NumCol", "group_by_column": "CatCol", "operation": "sum/avg/count"}},
          "sort_by": {{"column": "ColumnName", "ascending": true/false}},
          "chart_suggestions": {{ "chart_type": "bar/pie/line/table", "x_axis": "XCol", "y_axis": "YCol", "title": "Chart Title"}},
          "narrative_summary": "Your detailed data findings summary."
        }}
        If the question is abstract, out-of-scope (like 'in India' if data isn't from India), 
        or cannot be answered with the data, provide this structure:
        {{
          "filters": [], "analysis_type": "text_only", "aggregation": null, "sort_by": null,
          "chart_suggestions": {{ "chart_type": "none", "x_axis": null, "y_axis": null, "title": "Info"}},
          "narrative_summary": "Your helpful text answer OR 'Please ask a query relevant to the uploaded data.'"
        }}
        ALWAYS return one of these two valid JSON structures.
    """
    
    response_text = "AI response was not generated."
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        response_text = response.text

        json_match = re.search(r"```json\s*([\s\S]+?)\s*```", response_text, re.IGNORECASE)
        if json_match:
            response_text = json_match.group(1).strip()
        else:
            response_text = response_text.strip()

        llm_response_json = json.loads(response_text)

        # <<<< CHANGE: Start Try/Except for structure errors >>>>
        try:
            # Check if we got a 'none' chart type - means AI says it's text only
            chart_suggestions = llm_response_json.get("chart_suggestions", {})
            if chart_suggestions.get("chart_type") == "none":
                return jsonify({
                    "llm_suggestion": llm_response_json,
                    "processed_data": [],
                    "data_columns": [],
                    "narrative_summary": llm_response_json.get("narrative_summary", "Please ask a query relevant to the data.")
                })

            # If not 'none', proceed with data processing (it might still fail if structure is bad)
            processed_df = df.copy()

            # Apply Filters
            filters = llm_response_json.get("filters", [])
            if filters:
                for f_dict in filters:
                    col, op, val = f_dict.get("column"), f_dict.get("operator"), f_dict.get("value")
                    if col and op and val is not None and col in processed_df.columns:
                        # ... (Your filter logic as before) ...
                        try:
                            col_dtype = processed_df[col].dtype
                            if pd.api.types.is_numeric_dtype(col_dtype):
                                val_num = float(val)
                                if op == "equals": processed_df = processed_df[processed_df[col] == val_num]
                                elif op == "greater_than": processed_df = processed_df[processed_df[col] > val_num]
                                elif op == "less_than": processed_df = processed_df[processed_df[col] < val_num]
                                else: processed_df = processed_df[processed_df[col].astype(str).str.contains(str(val), case=False, na=False)]
                            else: 
                                val_str = str(val)
                                if op == "equals": processed_df = processed_df[processed_df[col].astype(str) == val_str]
                                elif op == "contains": processed_df = processed_df[processed_df[col].astype(str).str.contains(val_str, case=False, na=False)]
                        except Exception as filter_e: print(f"Filter error: {filter_e}")

            # Apply Aggregation
            final_columns = processed_df.columns.tolist()
            aggregation_config = llm_response_json.get("aggregation")
            if llm_response_json.get("analysis_type") == "aggregation" and aggregation_config:
                # ... (Your aggregation logic as before) ...
                group_col = aggregation_config.get("group_by_column")
                agg_col = aggregation_config.get("column_to_aggregate")
                op = aggregation_config.get("operation")
                if group_col and agg_col and op and group_col in processed_df.columns and agg_col in processed_df.columns:
                     try:
                        if op == "sum": temp_df = processed_df.groupby(group_col, as_index=False)[agg_col].sum()
                        elif op == "average": temp_df = processed_df.groupby(group_col, as_index=False)[agg_col].mean()
                        elif op == "count": temp_df = processed_df.groupby(group_col, as_index=False)[agg_col].count()
                        else: temp_df = processed_df
                        new_col_name = f"{agg_col}_{op}"
                        temp_df = temp_df.rename(columns={agg_col: new_col_name})
                        processed_df = temp_df
                        final_columns = processed_df.columns.tolist()
                     except Exception as agg_e: print(f"Aggregation Error: {agg_e}")

            # Apply Sorting
            sort_config = llm_response_json.get("sort_by")
            if sort_config and sort_config.get("column"):
                # ... (Your sorting logic as before) ...
                sort_col = sort_config.get("column")
                if sort_col in processed_df.columns:
                    processed_df = processed_df.sort_values(by=sort_col, ascending=sort_config.get("ascending", True))


            processed_df_safe = processed_df.head(50).where(pd.notnull(processed_df.head(50)), None)
            chart_data = processed_df_safe.to_dict(orient='records')

            return jsonify({
                "llm_suggestion": llm_response_json,
                "processed_data": chart_data,
                "data_columns": final_columns,
                "narrative_summary": llm_response_json.get("narrative_summary", "Analysis complete.")
            })

        # <<<< CHANGE: Catch block for structure errors >>>>
        except (KeyError, TypeError, AttributeError) as struct_err:
            print(f"Structure Error processing LLM response: {struct_err}")
            # The LLM gave JSON, but not the expected structure. Return its summary as text.
            return jsonify({
                    "llm_suggestion": {"chart_suggestions": {"chart_type": "none"}},
                    "processed_data": [],
                    "data_columns": [],
                    "narrative_summary": llm_response_json.get("narrative_summary", "Could not process AI response structure. Please rephrase.")
                })

    except json.JSONDecodeError as json_err:
        print(f"JSON Decode Error: {json_err}")
        return jsonify({"error": f"AI response not valid JSON. Raw: {response_text[:200]}..."}), 500
    except Exception as e:
        print(f"General Error: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)