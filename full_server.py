from flask import Flask, request, jsonify, render_template
from chat_jais import get_jais_response
from chat_llama import get_llama_response

app = Flask(__name__)

# run on llama

@app.route("/", methods=["GET"])
def home():
    return render_template("page.html")


@app.route("/get_llama_actions", methods=["POST"])
def get_llama_actions():

    try:
        crisis = request.json.get("crisis")
        sectors = request.json.get("sectors")
        is_injuries = request.json.get("is_injuries")

        # translate from user lang to system lang
        # ...

        # check if crisis is fake or not

        response = {'data':{}}

        for sector in sectors:
            
            result = get_llama_response(crisis= crisis, sector= sector, is_injuries=is_injuries)

            response["data"][sector] = result


        # translate from system lang to user lang
        # ...
        
        #return render_template("page.html", output=response)
        return response
    
    except Exception as e:
    
        print(f"An exception occurred: {e}")

        return {'data': None }

# run on jais
@app.route("/get_jais_actions", methods=["POST"])
def get_jais_actions():

    try:
        crisis = request.json.get("crisis")
        sectors = request.json.get("sectors")
        is_injuries = request.json.get("is_injuries")

        # translate from user lang to system lang 
        # ...

        # check if crisis is fake or not

        response = {'data':{}}

        for sector in sectors:
            
            result = get_jais_response(crisis= crisis, sector= sector,is_injuries=is_injuries )

            response["data"][sector] = result


        # translate from system lang to user lang
        # ...
        #return render_template("page.html", output=response)
        return response
    
    except Exception as e:
    
        print(f"An exception occurred: {e}")

        return {'data': None }
@app.route("/run-model", methods=["POST"]) 
def get_form_data():
    try:
        crisis = request.json.get("crisis")
        sectors = request.json.get("sectors")
        is_injuries = request.json.get("is_injuries")
        model = request.json.get("model")

        response = {'data':{}}

        if model == "jais":
            
            response = get_jais_response(crisis= crisis, sector= sector,is_injuries=is_injuries )
            return render_template("page.html", output=response)
        
        elif model == "llama":
                
            response = get_llama_response(crisis= crisis, sector= sector, is_injuries=is_injuries)
               
            return render_template("page.html", output=response)
                
        #return jsonify({"error":"error occured}),400


    
    except Exception as e:
    
        print(f"An exception occurred: {e}")

       # return jsonify({"error":"error occured}),400








if __name__ == "__main__":

    app.run(port=6829)
