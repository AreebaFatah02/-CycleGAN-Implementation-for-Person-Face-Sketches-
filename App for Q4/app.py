import torch
import torch.nn as nn
# AREEBA FATAH
# 21I-0349
# AI-DS GENAI
# QUESTION 4 ASSIGNMENT 4

# ----------> STEP 1: IMPORTING THE REQUIRED LIBRARIES
from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
from torchvision import transforms
import io
import base64


# ------------------> STEP 2.1: GENERATOR SKETCH TO PHOTO 
class GeneratorSketchToPhoto(nn.Module):
    def __init__(self):
        super(GeneratorSketchToPhoto,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,3,kernel_size=7,stride=1,padding=3),
            nn.Tanh()
        )

    def forward(self,x):
        return self.model(x)


#------------------> STEP 2.2: GENERATOR FOR PHOTO TO SKETCH
class GeneratorPhotoToSketch(nn.Module):
    def __init__(self):
        super(GeneratorPhotoToSketch,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,1,kernel_size=7,stride=1,padding=3),
            nn.Tanh()
        )

    def forward(self,x):
        return self.model(x)


# ------------------> STEP 3: LOADING THE SAVED MODELS (WEIGHTS)
G_XY=GeneratorSketchToPhoto()
G_YX=GeneratorPhotoToSketch()
G_XY.load_state_dict(torch.load('generator_sketch_to_photo.pth',map_location='cpu',weights_only=True))
G_YX.load_state_dict(torch.load('generator_photo_to_sketch.pth',map_location='cpu',weights_only=True))
G_XY.eval()
G_YX.eval()


#------------------> STEP 4: FLASK APP
app=Flask(__name__)
transform_sketch=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
transform_photo=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
inv_transform=transforms.Compose([
    transforms.Normalize([-1],[2]),
    transforms.ToPILImage()
])


#------------------> STEP 5: DEFINING PAGE ROUTES
@app.route('/')
def index():
    return render_template('index.html')

# FOR DECIDING THE ALLOWED FILES
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in {'png','jpg','jpeg'}


# ------------------> STEP 5.1 : FOR LIVE CAMERA ROUTES
@app.route('/convert/live',methods=['POST'])
def convert_live():
    data=request.json
    image_data=data.get('image')
    conversion_type=data.get('conversion')
    image=Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    if conversion_type=='photo-to-sketch':
        image=transform_photo(image).unsqueeze(0)
        model=G_YX
    else:
        image=transform_sketch(image.convert('L')).unsqueeze(0)
        model=G_XY
    with torch.no_grad():
        generated=model(image)
    generated=generated.squeeze(0)
    result_image=inv_transform(generated)
    buffer=io.BytesIO()
    result_image.save(buffer,format='JPEG')
    buffer.seek(0)
    img_str=base64.b64encode(buffer.read()).decode()
    return jsonify({'image':f'data:image/jpeg;base64,{img_str}'})


# ------------------> STEP 5.2: FOR SKETCH TO PHOTO ROUTE
@app.route('/convert/sketch-to-photo',methods=['POST'])
def convert_sketch_to_photo():
    if 'file' not in request.files:
        return jsonify({'error':'No file provided'}),400
    file=request.files['file']
    if allowed_file(file.filename):
        image=Image.open(file).convert('L')
        image=transform_sketch(image).unsqueeze(0)
        with torch.no_grad():
            generated_photo=G_XY(image)
        generated_photo=inv_transform(generated_photo.squeeze(0))
        img_io=io.BytesIO()
        generated_photo.save(img_io,'JPEG')
        img_io.seek(0)
        return send_file(img_io,mimetype='image/jpeg')

#------------------> STEP 5.3: FOR PHOTO TO SKETCH ROUTE
@app.route('/convert/photo-to-sketch',methods=['POST'])
def convert_photo_to_sketch():
    if 'file' not in request.files:
        return jsonify({'error':'No file provided'}),400
    file=request.files['file']
    if allowed_file(file.filename):
        image=Image.open(file).convert('RGB')
        image=transform_photo(image).unsqueeze(0)
        with torch.no_grad():
            generated_sketch=G_YX(image)
        generated_sketch=inv_transform(generated_sketch.squeeze(0))
        img_io=io.BytesIO()
        generated_sketch.save(img_io,'JPEG')
        img_io.seek(0)
        return send_file(img_io,mimetype='image/jpeg')


# ------------------> STEP 6: CALLING THE MAIN APP
if __name__=='__main__':
    app.run(debug=True)
