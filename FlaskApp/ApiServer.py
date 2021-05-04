"""Face Swap Flask Application. Using CSS/template from
https://us-west-2-tcdev.s3.amazonaws.com/courses/AWS-100-ADG/v1.1.0/exercises/ex-s3-upload.zip"""
# Original template/CSS copyright:
# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except
# in compliance with the License. A copy of the License is located at
#
# https://aws.amazon.com/apache-2-0/
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from flask import Flask, render_template_string, request, send_file
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
from wtforms import StringField, SubmitField, BooleanField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap

from nats.aio.client import Client as NATS
from minio import Minio
from minio.error import S3Error

import asyncio
import pickle
import cv2
import re
import util
import io
import os

# Flask setup
application = Flask(__name__)
application.secret_key = util.random_hex_bytes(5)
application.config["UPLOAD_FOLDER"] = "."
bootstrap = Bootstrap(application)
application.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# Flask forms
class PhotoForm(FlaskForm):
    """flask_wtf form class the file upload"""
    photo = FileField('image', validators=[
        FileRequired()
    ])


class NameForm(FlaskForm):
    name = StringField("Face name", validators=[DataRequired()])


class SwapForm(FlaskForm):
    base = StringField("Base face's name", validators=[DataRequired()])
    donor = StringField("Donor face's name", validators=[DataRequired()])
    swap = SubmitField("Swap faces!")


class RatingForm(FlaskForm):
    base = StringField("Base face's name", validators=[DataRequired()])
    donor = StringField("Donor face's name", validators=[DataRequired()])
    liked = BooleanField("Like photo")
    submit = SubmitField("Submit rating")


# NATS async functions
async def start(loop):
    await nc.connect("127.0.0.1:4222", loop=loop)


async def stop():
    await nc.close()


async def run(topic, message):
    try:
        response = await nc.request(topic, message.encode("utf-8"), timeout=10)
        return response.data
    except Exception as e:
        print("Error:", e)


# CRUD helper functions
def get_face(name):
    data = get_object(name, "faces")
    if type(data) == bool and not data:
        return None
    else:
        return pickle.loads(get_object(name, "faces"))


def get_object(name, bucket_name):
    try:
        response = client_minio.get_object(buckets[bucket_name], name)
        data = response.data
        response.close()
        response.release_conn()
        return data
    except S3Error as e:
        if e.code == "NoSuchKey":
            print("NoSuchKey in get_face")
            return False  # TODO: return None instead?
        else:
            raise e


def set_object(name, bucket_name, object_source):
    object_pickled = pickle.dumps(object_source)
    client_minio.put_object(buckets[bucket_name], name, io.BytesIO(object_pickled), len(object_pickled))


# Flask routes
@application.route("/", methods=("GET",))
def home():
    return render_template_string("""
                {% extends "main.html" %}
                {% block content %}
                <h4>Cloud Native Face Swapping</h4>
                <p>
                Welcome to our face swapping project for ECGR 4090! Click a label in the navigation bar to start using
                the application. Each label in the navigational bar is also exposed as a REST API.
                </p>
                <p>
                Minio: http://127.0.0.1:9000/minio/face-images/
                </p>
                <p>
                Neo4j: http://localhost:7474/browser/
                </p>
                {% endblock %}
                    """)


@application.route("/create", methods=('GET', 'POST'))
def create():
    """Create Image in Database"""
    form = PhotoForm()
    message = None
    if form.validate_on_submit():
        filename = secure_filename(form.photo.data.filename)
        name = filename[:filename.find(".")]  # removes .jpg or whatever is the extension
        image_bytes = util.resize_image(form.photo.data, (300, 300))
        if image_bytes:
            set_object(name, "faces", image_bytes)
            message = loop.run_until_complete(run("create", name)).decode("utf-8")

    return render_template_string("""
            {% extends "main.html" %}
            {% block content %}
            <h3>Upload Photo</h3>
            <form method="POST" enctype="multipart/form-data" action="{{ url_for('create') }}">
                {{ form.csrf_token }}
                  <div class="control-group">
                   <label class="control-label">Photo</label>
                    {{ form.photo() }}
                  </div>

                   <div class="control-group">
                    <div class="controls">
                        <input class="btn btn-primary" type="submit" value="Upload">
                    </div>
                  </div>
            </form>

            {% if message %}
            <p>{{message}}</p>
            {% endif %}

            {% endblock %}
                """, form=form, message=message)


@application.route("/swap", methods=('GET', 'POST'))
def swap():
    """Supply face names, get back a facial image"""
    form = SwapForm()
    list_nodes = re.sub(pattern="[\[\{\':\]\}]", repl="",
                        string=loop.run_until_complete(run("list", "")).decode("utf-8")
                        .replace("Node", "")
                        .replace("name", ""))
    if len(request.args) != 0:
        base = request.args["base"]
        donor = request.args["donor"]
        swap_name_input = donor + "|" + base
        result = loop.run_until_complete(run("swap", swap_name_input))
        print(result)
        if result is not None and result.decode("utf-8") == swap_name_input:
            return send_file(
                io.BytesIO(cv2.imencode(".jpg", get_face(swap_name_input))[1]),
                mimetype='image/jpeg',
                as_attachment=True,
                attachment_filename=swap_name_input + '.jpg')
        else:
            return "failed"
    elif form.validate_on_submit():
        print("Base name:", form.base.data, "Donor name:", form.donor.data)
        swap_name_input = form.donor.data + "|" + form.base.data
        swap_name = loop.run_until_complete(run("swap", swap_name_input))

        if swap_name is not None and swap_name != "":
            print("Swap name is", swap_name)
            face = get_face(swap_name_input)
            photo = "swap.jpg"
            cv2.imwrite("static/" + photo, face)
            client_minio.remove_object(buckets["faces"], swap_name_input)
            return render_template_string("""
                            {% extends "main.html" %}
                            {% import "bootstrap/wtf.html" as wtf %}
                            {% block content %}
                            <h3>Swap Faces</h3>
                            <h4>Valid Names</h4>
                            <p>{{list}}</p>

                            <hr/>
                            <h3>Photo</h3>
                            <img width="500" src="/static/{{photo}}" />
                            {{ wtf.quick_form(form) }}

                            {% endblock %}
                                """, form=form, photo=photo, list=list_nodes)
        else:
            return "Error: no result"
    else:
        return render_template_string("""
                {% extends "main.html" %}
                {% import "bootstrap/wtf.html" as wtf %}
                {% block content %}
                <h3>Swap Faces</h3>
                <h4>Valid Names</h4>
                <p>{{list}}</p>
                {{ wtf.quick_form(form) }}
                {% endblock %}
                    """, form=form, list=list_nodes)


@application.route("/rate", methods=("GET", "POST"))
def rate():
    rating = RatingForm()
    list_nodes = re.sub(pattern="[\[\{\':\]\}]", repl="",
                        string=loop.run_until_complete(run("list", "")).decode("utf-8").replace("Node", "")
                        .replace("name", ""))
    print(request.args)
    if len(request.args) != 0:
        base = request.args["base"]
        donor = request.args["donor"]
        rating_arg = request.args["liked"]
        return str(loop.run_until_complete(run("rank", donor + "|" + base + "|" + rating_arg)))
    elif rating.validate_on_submit():
        print("Rating", 1 if rating.liked.data else 0)
        base = rating.base.data
        donor = rating.donor.data
        rating_arg = str(1 if rating.liked.data else 0)
        rating = str(loop.run_until_complete(run("rank", donor + "|" + base + "|" + rating_arg)))
        print(donor, base, "rating is", rating)
        return render_template_string("""
                                {% extends "main.html" %}
                                {% import "bootstrap/wtf.html" as wtf %}
                                {% block content %}
                                <h3>Rank Faces</h3>
                                <h4>Valid Names</h4>
                                <p>{{list}}</p>
                                <p>{{rating}}</p>
                                {{ wtf.quick_form(form) }}
                                {{rank}}
                                {% endblock %}
                                    """, form=rating, list=list_nodes, rating=rating)
    else:
        return render_template_string("""
                        {% extends "main.html" %}
                        {% import "bootstrap/wtf.html" as wtf %}
                        {% block content %}
                        <h3>Rank Faces</h3>
                        <h4>Valid Names</h4>
                        <p>{{list}}</p>
                        {{ wtf.quick_form(form) }}
                        {{rank}}
                        {% endblock %}
                            """, form=rating, list=list_nodes)


@application.route("/list", methods=('GET',))
def list_faces():
    list_nodes = loop.run_until_complete(run("list", ""))
    return list_nodes.decode("utf-8")


@application.route("/delete", methods=("GET", "POST"))
def delete():
    """Delete Image from Databases"""
    form = NameForm()
    message = None
    if len(request.args) != 0:
        name = request.args["name"]
        return loop.run_until_complete(run("delete", name))
    elif form.validate_on_submit():
        name = form.name.data
        print("Removing", name)
        message = loop.run_until_complete(run("delete", name)).decode("utf-8")
        # return render_template_string("""
        #                 {% extends "main.html" %}
        #                 {% import "bootstrap/wtf.html" as wtf %}
        #                 {% block content %}
        #                 {{ wtf.quick_form(form) }}
        #                 <p>{{message}}</p>
        #                 {% endblock %}
        #                     """, form=form, message=message.decode("utf-8"))

    list_nodes = re.sub(pattern="[\[\{\':\]\}]", repl="",
                        string=loop.run_until_complete(run("list", "")).decode("utf-8").replace("Node", "")
                        .replace("name", ""))

    return render_template_string("""
                        {% extends "main.html" %}
                        {% import "bootstrap/wtf.html" as wtf %}
                        {% block content %}
                        <h3>Delete a Face</h3>
                        <h4>Valid Names</h4>
                        <p>{{list}}</p>
                        {{ wtf.quick_form(form) }}
                        {% if message %}
                        <p>{{message}}</p>
                        {% endif %}
                        {% endblock %}
                    """, form=form, list=list_nodes, message=message)


@application.route("/view", methods=("GET", "POST"))
def view():
    """Supply a faces name, get back an image"""
    form = NameForm()
    url = None
    photo = None
    rating = None
    if len(request.args) != 0:
        name = request.args["name"]
        result = get_face(name)
        if result is not None or result != b'':
            return send_file(
                io.BytesIO(result),
                mimetype='image/jpeg',
                as_attachment=True,
                attachment_filename=name + '.jpg')
        else:
            return "failed"
    elif form.validate_on_submit():
        name = form.name.data
        print("Viewing", name)
        try:
            os.remove("static/view.jpg")
        except FileNotFoundError:
            pass  # No problems here, just means it doesn't exist
        rating = loop.run_until_complete(run("view", name)).decode()
        face = get_face(name)
        if rating is not None and rating != "" and face is not None:
            print("Swap name is", name)
            photo = "view.jpg"
            with open("static/" + photo, "wb") as file:
                file.write(face)
        else:
            return "Error: no result"

    list_nodes = re.sub(pattern="[\[\{\':\]\}]", repl="",
                        string=loop.run_until_complete(run("list", "")).decode("utf-8").replace("Node", "")
                        .replace("name", ""))

    return render_template_string("""
                    {% extends "main.html" %}
                    {% import "bootstrap/wtf.html" as wtf %}
                    {% block content %}
                    <h3>View Face</h3>
                    <h4>Valid Names</h4>
                    <p>{{list}}</p>
                    {{ wtf.quick_form(form) }}

                    {% if photo %}
                    <h3>Photo</h3>
                    <h4>Photo Rating</h4>
                    <p>{{rating}}</p>
                    <h4>Image</h4>
                    <img width="500" src="/static/{{photo}}" />
                    {% endif %}

                    {% endblock %}
                        """, form=form, url=url, photo=photo, list=list_nodes, rating=rating)


@application.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    nc = NATS()
    loop.run_until_complete(start(loop))

    client_minio = Minio(
        "localhost:9000",
        access_key="AKIAIOSFODNN7EXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        secure=False
    )
    buckets = {"faces": "face-images", "points": "facial-points"}
    for bucket in buckets.values():
        found = client_minio.bucket_exists(bucket)
        if not found:
            client_minio.make_bucket(bucket)
        else:
            print("Bucket '{}' already exists".format(bucket))

    # http://flask.pocoo.org/docs/0.12/errorhandling/#working-with-debuggers
    use_c9_debugger = False
    application.run(use_debugger=not use_c9_debugger, debug=True,
                    use_reloader=not use_c9_debugger, host='0.0.0.0', port=8080)

    loop.run_until_complete(stop())
    loop.close()
