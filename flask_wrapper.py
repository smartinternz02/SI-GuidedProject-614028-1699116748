import os
from PIL import Image

from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from const import IMAGE_SIZE, MAX_LENGTH, UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from image import load_features_from_img
from model_helper import (
    ModelName,
    load_captioning_model,
    predict_caption,
)

from caption import load_tokenizer

print("Loadin the Captioning Model....")
model = load_captioning_model(ModelName.EARLY_STOPPED_MODEL)
print("Loadin the tokenizer....")
tokenizer = load_tokenizer()


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SECRET_KEY"] = "secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)


@app.route("/")
def home():
    if "username" in session:
        return render_template("index.html")
    return redirect(url_for("signin"))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=["POST"])
def predict_from_image_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in the form!")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected!")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # get the abs path of the file
            current_folder_path = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(
                current_folder_path, app.config["UPLOAD_FOLDER"], filename
            )
            file.save(file_path)
            print("File saved at: ", file_path)
            image = Image.open(file_path)
            img_features = load_features_from_img(image, IMAGE_SIZE)
            caption = predict_caption(model, img_features, tokenizer, MAX_LENGTH)
            return render_template("index.html", caption=caption, image_path=filename)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        user_name = request.form.get("username", None)
        password = request.form.get("password", None)
        if user_name is None and password is None:
            return "Please enter your username and password", 400
        user = User.query.filter_by(username=user_name).first()
        if user:
            return "User already exists", 400
        hashed_password = generate_password_hash(password)
        new_user = User(username=user_name, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("signin"))
    return render_template("signup.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    user_name = request.form.get("username", None)
    password = request.form.get("password", None)
    msg = None
    if request.method == "POST":
        if user_name is None and password is None:
            msg = "Please enter your username and password"
        else:
            user = User.query.filter_by(username=user_name).first()
            if user is None:
                msg = "User doesn't exist"
            else:
                password = request.form["password"]
                is_valid_password = check_password_hash(user.password, password)
                if user and is_valid_password:
                    session["username"] = user.username
                    return redirect(url_for("home"))
    return render_template("signin.html", msg=msg)


if __name__ == "__main__":
    print("Setting up the database ....")
    with app.app_context():
        db.create_all()
    print("Starting the server in [ dev-mode ] ....")
    app.run(debug=True)
