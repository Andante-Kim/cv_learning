import os, sys
import numpy as np
import cv2
from flask import Flask, flash, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

from .img_processing import embossing, haar_face, yolo_v3, mask_rcnn

def create_app():
    app = Flask(__name__)

    app.secret_key = os.urandom(24)
    app.config['RESULT_FOLDER'] = 'result_images'   # 반드시 폴더 미리 생성
    app.config['UPLOAD_FOLDER'] = 'uploads'         # 반드시 폴더 미리 생성

    @app.route('/upload_img/<filename>')
    def upload_img(filename):		# 예, http://127.0.0.1:5000/upload_img/eagle.jpg
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/result_img/<filename>')
    def result_img(filename):
        return send_from_directory(app.config['RESULT_FOLDER'], filename)

    @app.route('/img_result', methods=['GET', 'POST'])
    def img_result():
        if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']
            style = request.form.get('style')
            print(style)

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            print(basepath)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename)) # 예, C:\....\pybo\uploads\eagle.jpg
            f.save(file_path)
            file_name = os.path.basename(file_path)	# 예, eagle.jpg

            # reading the uploaded image
            img = cv2.imread(file_path)

            if style == "Embossing":
                processed = embossing(img)

                result_fname = os.path.splitext(file_name)[0] + "_embossing.jpg"		# 예, eagle_embossing.jpg
                result_path = os.path.join(basepath, 'result_images', secure_filename(result_fname))
                fname = os.path.basename(result_path)
                print(fname)
                cv2.imwrite(result_path, processed)
                return render_template('img_result.html', file_name=file_name, result_file=fname)
            elif style == "Stylization":
                processed = cv2.stylization(img,sigma_s=60,sigma_r=0.45)

                result_fname = os.path.splitext(file_name)[0] + "_stylization.jpg"
                result_path = os.path.join(basepath, 'result_images', secure_filename(result_fname))
                fname = os.path.basename(result_path)
                cv2.imwrite(result_path, processed)
                return render_template('img_result.html', file_name=file_name, result_file=fname)
            elif style == "PencilGray":
                processed, _ = cv2.pencilSketch(img,sigma_s=60,sigma_r=0.07,shade_factor=0.02)

                result_fname = os.path.splitext(file_name)[0] + "_pencil_gray.jpg"
                result_path = os.path.join(basepath, 'result_images', secure_filename(result_fname))
                fname = os.path.basename(result_path)
                cv2.imwrite(result_path, processed)
                return render_template('img_result.html', file_name=file_name, result_file=fname)
            elif style == "PencilColor":
                _, processed = cv2.pencilSketch(img,sigma_s=60,sigma_r=0.07,shade_factor=0.02)

                result_fname = os.path.splitext(file_name)[0] + "_pencil_color.jpg"
                result_path = os.path.join(basepath, 'result_images', secure_filename(result_fname))
                fname = os.path.basename(result_path)
                cv2.imwrite(result_path, processed)
                return render_template('img_result.html', file_name=file_name, result_file=fname)
            elif style == "OilPainting":
                processed = cv2.xphoto.oilPainting(img, 10, 1, cv2.COLOR_BGR2Lab)

                result_fname = os.path.splitext(file_name)[0] + "_oil.jpg"
                result_path = os.path.join(basepath, 'result_images', secure_filename(result_fname))
                fname = os.path.basename(result_path)
                cv2.imwrite(result_path, processed)
                return render_template('img_result.html', file_name=file_name, result_file=fname)
            elif style == "EdgePreserving":
                processed = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)

                result_fname = os.path.splitext(file_name)[0] + "_edgePresv.jpg"
                result_path = os.path.join(basepath, 'result_images', secure_filename(result_fname))
                fname = os.path.basename(result_path)
                cv2.imwrite(result_path, processed)
                return render_template('img_result.html', file_name=file_name, result_file=fname)
            elif style == "FaceDetection":
                processed = haar_face(img, basepath+"/data//")

                result_fname = os.path.splitext(file_name)[0] + "_face.jpg"
                result_path = os.path.join(basepath, 'result_images', secure_filename(result_fname))
                fname = os.path.basename(result_path)
                cv2.imwrite(result_path, processed)
                return render_template('img_result.html', file_name=file_name, result_file=fname)
            elif style == "Yolo":
                processed = yolo_v3(img, basepath+"/data//")

                result_fname = os.path.splitext(file_name)[0] + "_yolo.jpg"
                result_path = os.path.join(basepath, 'result_images', secure_filename(result_fname))
                fname = os.path.basename(result_path)
                cv2.imwrite(result_path, processed)
                return render_template('img_result.html', file_name=file_name, result_file=fname)
            elif style == "Segmentation":
                processed = mask_rcnn(img, basepath+"/data//")

                result_fname = os.path.splitext(file_name)[0] + "_segmentation.jpg"
                result_path = os.path.join(basepath, 'result_images', secure_filename(result_fname))
                fname = os.path.basename(result_path)
                cv2.imwrite(result_path, processed)
                return render_template('img_result.html', file_name=file_name, result_file=fname)

            else:
                flash('Please select style')
                return render_template('img_processing.html')

        return ""

    @app.route('/')
    #def helo_pybo():
    #    return 'Hello__ Pybo'
    def index():
        return render_template('index.html')

    return app
