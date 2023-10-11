#CODE IMPORTS
import openai
import requests
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageFilter
import random
import numpy as np
from threading import Thread
import threading

#THIS PROGRAM: Takes inputs, uses the class AIImage to manage generating AI Instagram/Youtube ad images
#Use the generate method to generate the images (from DALLE) and then the design method to output x number of designs
#This code is not finished and requires more specific designs and consistent generation

#THE MAIN CLASS TO GENERATE IMAGES
class AIImage:
    def __init__(self,api_key="",text="",subText="",logo="",colors=[],description="",size="",target="",font="arial.ttf",image="",image_type=""):
        #Constructor Class
        #DEFINE SOME CONSTANTS HERE
        self.GPT_SIZE="1024x1024"
        self.CHROMA=0.5#Percentage of images that are chroma
        self.DESIGN_OPTIONS={"fill":0.2,"blur":0.15,"circle":0.1,"none":0.05}
        #ADD CORNER OPTION!
        self.TEXT_STYLES={"fill":0.4,"fill-sub":0.3,"flat":0.2,"flat-sub":0.1}


        self.logo=logo
        self.colors=colors.split(",")#Add black and white to this
        for i in range(len(self.colors)):
            self.colors[i]=hex_to_rgb(self.colors[i].replace("#",""))
        self.colors=self.colors+[(0,0,0),(255,255,255)]#Add these as default
        self.description=description
        self.size=size # Smaller than 1024x1024, then just scale it!
        self.target=target
        self.text=text
        self.subText=subText
        self.font=font
        self.generated_images=[image]# Store image in the generated array
        self.image_type=image_type

        if self.image_type=="Instagram":
            self.size=(1080,566)
            self.DESIGN_OPTIONS={"fill":0.2,"blur":0.15,"circle":0.1,"none":0.05}
            self.CHROMA=1
        elif self.image_type=="Youtube":
            self.size=(1280,720)
            self.DESIGN_OPTIONS={"fill":0.2,"blur":0.15,"circle":0.05,"none":0.1}
            self.CHROMA=0.1


        self.api_key=api_key
        
        openai.api_key = self.api_key #"PUT KEY HERE"

        self.designs=[]


        self.lock=threading.Lock()

    #############################################GPT CODE

    #GENERATE IMAGE PROMPTS USING CHATGPT
    def generate_image_prompt(self):
        if random.random()<self.CHROMA:#If a filled background
            text=chatGPT("Give me a single noun (object) about: "+self.description)
            return "Realistic "+str(text)+" "+str(random.choice(self.colors))+" background"
        else:
            text=chatGPT("Give me a very short prompt for an AI image for: "+self.description)
            return str(text)+" in the style of a hyperrealistic photograph"

    #GENERATE A SINGLE IMAGE FROM OPENAI
    def generate_image(self):
        prompt=self.generate_image_prompt()
        response = openai.Image.create(prompt=prompt,n=1,size=self.GPT_SIZE)#Generate Images Here
        size=max(self.size[0],self.size[1])
        img=requests.get(response["data"][0]["url"],stream=True)
        img=PIL.Image.open(img.raw)
        img=img.convert("RGBA")
        img=img.resize((size,size))
        img=img.crop(((size-self.size[0])/2,(size-self.size[1])/2,(size-self.size[0])/2+self.size[0],(size-self.size[1])/2+self.size[1]))
        #Creates an array of images
        self.lock.acquire()
        self.generated_images.append(img)
        self.lock.release()

    #GENERATE TEXT FROM CHATGPT
    def generate_text(self):
        if self.text=="":#Only do if none given
            return chatGPT("Generate a 2-8 word Title for: "+self.description)
        return self.text

    #GENERATE SUBTEXT FROM CHATGPT
    def generate_subText(self):#(subText)
        if self.subText=="":#Only do if none given
            return chatGPT("Generate a 4-10 word Sub-Title for: "+self.description)
        return self.subText

    ####################################################
    #GENERATES THE IMAGES USING THREADING SO FASTER
    def generate_images(self,image_count):
        self.generated_images=[]
        threads=[]
        for i in range(image_count):
            t = Thread(target=self.generate_image)#Call function for each
            threads.append(t)
        # start the threads
        [ t.start() for t in threads ]
        # wait for the threads to finish
        [ t.join() for t in threads ]
        return

    #RETURNS THE GIVEN NUMBER OF IMAGE DESIGNS
    def get_designs(self,design_count):
        if self.generated_images==[""]:
            print("ERROR: You need to pass and image or generate images before designs are possible")
            return 
        else:
            self.designs=[]#Reset the designs
            #IMPLEMENT THEADING HERE
            threads = []
            for i in range(design_count):
                #Divide all ids by number of threads
                # print(dir(self.generate_images))
                t = Thread(target=self.design(random.choice(self.generated_images)))#Call function for each
                threads.append(t)
            # start the threads
            [ t.start() for t in threads ]
            # wait for the threads to finish
            [ t.join() for t in threads ]
            return self.designs


    #ADDS DESIGNS AND TEXT TO THE AI GENERATED IMAGE    
    def design(self, img):
        img = img.copy()
        colors = self.colors.copy()
        Idraw = PIL.ImageDraw.Draw(img, "RGBA")  # Create drawable surface
        # Pick a random design format based on given weights
        choice = random.choices(list(self.DESIGN_OPTIONS.keys()), weights=self.DESIGN_OPTIONS.values(), k=1)[0]

        colors, color = get_color(colors)
        # 1. APPLY DESIGN
        if check_chroma_possible(img, self.size[0], self.size[1]):
            Idraw, img, box = position(Idraw, img, self.size[0], self.size[1])
        else:
            if choice == "fill":
                img = color_fill(Idraw, img, color, self.size[0], self.size[1])
                Idraw = PIL.ImageDraw.Draw(img, "RGBA")  # Reinitialize Idraw
                if random.random() > 0.8:  # Randomly add a blur box
                    Idraw, img, box = blur_box(Idraw, img, int(self.size[0] * 0.7), int(self.size[1] * 0.7), self.size[0], self.size[1])
                else:
                    box = (self.size[0] * 0.1, self.size[1] * 0.1, self.size[0] - self.size[0] * 0.1, self.size[1] - self.size[1] * 0.1)
            elif choice == "blur":
                Idraw, img, box = blur_box(Idraw, img, int(self.size[0] * 0.7), int(self.size[1] * 0.7), self.size[0], self.size[1])
            elif choice == "circle":
                Idraw, img, box = circle(Idraw, img, color, self.size[0], self.size[1])
                if random.random() > 0.8:  # Randomly center text instead
                    box = (self.size[0] * 0.1, self.size[1] * 0.1, self.size[0] - self.size[0] * 0.1, self.size[1] - self.size[1] * 0.1)
            elif choice == "none":
                mult = random.uniform(0.2, 0.5)  # Random sized text box in the center
                box = (self.size[0] * mult, self.size[1] * mult, self.size[0] - self.size[0] * mult, self.size[1] - self.size[1] * mult)
        Idraw = PIL.ImageDraw.Draw(img, "RGBA")  # Reinitialize Idraw

        # 2. ADD THE LOGO
        Idraw, img = place_logo(Idraw, img, self.logo, self.size[0], self.size[1], box)

        colors, color = get_color(colors)
        # 3. ADD THE TEXT
        Idraw = add_text2(Idraw, img, self.generate_text(), self.generate_subText(), box, color, self.font)

        self.lock.acquire()
        self.designs.append(img)
        self.lock.release()
        return  # Return the designed image 
        























#CHECKS IF A ROW OF PIXELS IS ALL THE SAME COLOUR
def check_column(im_matrix,col,h,t):
    same=True
    prev=im_matrix[0][col]
    for pixel in range(50):
        color=im_matrix[pixel*(int(h/50))][col]
        if abs(int(color[0])-int(prev[0]))>t or abs(int(color[1])-int(prev[1]))>t or abs(int(color[2])-int(prev[2]))>t or abs(int(color[3])-int(prev[3]))>t:
            same=False
    return same,color
#CHECKS IF A ROW OF PIXELS IS ALL THE SAME COLOUR
def check_row(im_matrix,row,h,t):
    same=True
    prev=im_matrix[row][0]
    for pixel in range(50):
        color=im_matrix[row][pixel*(int(h/50))]
        if abs(int(color[0])-int(prev[0]))>t or abs(int(color[1])-int(prev[1]))>t or abs(int(color[2])-int(prev[2]))>t or abs(int(color[3])-int(prev[3]))>t:
            same=False
    return same,color
#RETURNS IF CHROMA POSSIBLE GIVEN THE IMAGE AND SIZE
def check_chroma_possible(img,w,h):
    im_matrix=np.array(img)#Create pixel array from image
    t=5
    same=True
    if w>=h:#If landscape/square
        for side in (0,w-1):#Check both sides are same solid color
            same,color=check_column(im_matrix,side,h,t)
        return same
    else:
        for side in (0,h-1):#Check both sides are same solid color
            same,color=check_row(im_matrix,side,w,t)
        return same











#MOVES THE IMAGE WHEN THE BACKGROUND IS CHROMA KEY (no background)
def position(Idraw,img,w,h):
    im_matrix=np.array(img)
    t=5#RGB threshold
    dist=40#Distance from item
    same=True
    if w>=h:#Landscape Image: Place Text at left or right
        align=random.choice(["right","left"])
        if align=="right":#This means we make space on the right
            c=0
            while same and c<w:#Find left edge
                same,color=check_column(im_matrix,c,h,t)
                c+=10
            if c<w*0.3:#If C is too small - set to half the image
                c=int(w/2)
            #Crop Image
            img=img.crop((c-dist,0,w,h))
            backCol=im_matrix[0][w-1]#Place Image on back
            back = PIL.Image.new(mode="RGBA",size=(w,h),color=(backCol[0],backCol[1],backCol[2],backCol[3]))
            back.paste(img,(0,0))
            #Space on the right will be ~ w-c
            box=((w-c)*random.uniform(1.02,1.1),h*random.uniform(0.1,0.3),(w)*random.uniform(0.9,0.98),h*random.uniform(0.7,0.9))
        else:
            c=w-1
            #Find Right edge
            while same and c>0:
                same,color=check_column(im_matrix,c,h,t)
                c-=10
            if c>w*(1-0.3):#If C is too small - set to half the image
                c=int(w/2)
            #Crop Image
            img=img.crop((0,0,c+dist,h))
            backCol=im_matrix[0][0]#Place Image on back
            back = PIL.Image.new(mode="RGBA",size=(w,h),color=(backCol[0],backCol[1],backCol[2],backCol[3]))
            back.paste(img,(c+dist,0))
            #Space on the right will be ~ c
            box=(w*random.uniform(0.02,0.1),h*random.uniform(0.1,0.3),c*random.uniform(0.9,0.98),h*random.uniform(0.7,0.9))
    else:#Portrait Image: Place Text at top or bottom
        align=random.choice(["top","bottom"])
        if align=="top":#Make Space at the TOP
            c=h-1
            while same and c>0:#Find bottom edge
                same,color=check_row(im_matrix,c,w,t)
                c-=10
            if c>h*(1-0.2):#If C is too small - set to half the image
                c=int(h/2)
            #Crop Image
            img=img.crop((0,0,w,c+dist))
            backCol=im_matrix[0][0]#Place Image on back
            back = PIL.Image.new(mode="RGBA",size=(w,h),color=(backCol[0],backCol[1],backCol[2],backCol[3]))
            back.paste(img,(0,c+dist))
            #Space on the top will be ~ c
            box=(w*random.uniform(0.1,0.2),h*random.uniform(0.02,0.01),w*random.uniform(0.8,0.9),c*random.uniform(0.9,0.98))
        else:#Make Space at the BOTTOM
            c=0
            while same and c<h:#Find bottom edge
                same,color=check_row(im_matrix,c,w,t)
                c+=10
            if c<h*0.2:#If C is too small - set to half the image
                c=int(h/2)
            #Crop Image
            img=img.crop((0,c-dist,w,h))
            backCol=im_matrix[0][0]#Place Image on back
            back = PIL.Image.new(mode="RGBA",size=(w,h),color=(backCol[0],backCol[1],backCol[2],backCol[3]))
            back.paste(img,(0,0))
            #Space on the top will be ~ c
            box=(w*random.uniform(0.1,0.2),(h-c)*random.uniform(1.02,1.1),w*random.uniform(0.8,0.9),h*random.uniform(0.9,0.98))
    return Idraw,back,box














#RANDOMLY PLACES A LOGO IMAGE THE MAIN IMAGE
def place_logo(Idraw,img,logo,w,h,box):
    #Load Logo and Initialise
    logo = PIL.Image.open(logo)
    logo = logo.convert("RGBA")
    logoW,logoH=logo.size
    x,y=0,0
    buf=10
    ratio=7#1/5 of the screen
    #Set Logo Size and Maintain Aspect Ratio
    if h<w:#Set width
        logoH=int(logoH*(int(w/ratio)/logoW))
        logoW=int(w/ratio)
    else:#Set Height
        logoW=int(logoW*(int(h/ratio)/logoH))
        logoH=int(h/ratio)
    logo=logo.resize((logoW,logoH))  
    d=max(logoW,logoH)*3#Diameter for circle if used
    #Select Corner, by getting quadrant text is near
    center=(box[0]+(box[2]-box[0])/2,box[1]+(box[3]-box[1])/2)
    if center[0]>int(w/2):#Right
        if center[1]>int(h/2):#Bottom
            x,y=w-logoW-buf,h-logoH-buf
            cx,cy=w-d/2,h-d/2
        else:#Top
            x,y=w-logoW-buf,buf
            cx,cy=w-d/2,-d/2
    else:#Left
        if center[1]>int(h/2):#Bottom
            x,y=buf,h-logoH-buf
            cx,cy=-d/2,h-d/2
        else:#Top
            x,y=buf,buf
            cx,cy=-d/2,-d/2
    #Ensure not over text
    if collide(box,(x,y,x+logoW,y+logoH)):#Check if overlapping text
        if x==buf:
            x=w-logoW-buf
        else:
            x=buf
        if y==buf:
            y=h-logoH-buf
        else:
            y=buf
    else:
        if random.random()>0.95:#Sometimes randomly place
            x=random.choice([buf,w-logoW-buf])
            y=random.choice([buf,h-logoH-buf])
        if random.random()>0.8:#Corner circle, only ever white
            Idraw.ellipse((cx,cy,cx+d,cy+d),fill=(255,255,255))
    img.paste(logo,(x,y),logo)#Place Logo
    return Idraw,img










#CREATES A COLOURED CIRCLE TO PLACE TEXT ON IN A RANDOM CORNER
def circle(Idraw,img,color,imgw,imgh):
        pos=max(imgw,imgh)
        top=left=-int(pos/2)*1.2
        #top=top+top*0.2
        bottom=right=pos+int(pos/2)
        r=int(max(imgw*1.2,imgh))
        #xmid=int(self.size[0]/2)
        ymid=int(imgh/2-r/2)
        box=(0,0,imgw,imgh)

        #Corners,TL,BL,BR,TR
        options=[(left,top,left+r,top+r),
                (left,bottom-r,left+r,bottom),
                (right-r,bottom-r,right,bottom),
                (right-r,top,right,top+r)]
        opt=random.choice(options)
        Idraw.ellipse(opt, fill=color)
        #Create text box bounds
        if opt==options[0] or opt==options[1]:#Left
            box1=20
            box3=imgw*0.5
        else:
            box1=imgw-imgw*0.5
            box3=imgw-20
        if opt==options[0] or opt==options[3]:
            box2=20
            box4=imgh*0.5
        else:
            box2=imgh-imgh*0.5
            box4=imgh-20
        box = (box1,box2,box3,box4)
        return Idraw,img,box















#PLACES TEXT IN A BOX USING A FILL ALGORITHM
def add_text2(Idraw,img,text,subText,box,color,font):
    #1. Start by splitting text!
    if random.random()>0.5:
        text=text.upper()
    text=text.split(" ")

    #Crazy Words Style
    rows=[]
    currentRow=""
    for i in text:
        if len(i)>10:
            if currentRow!="":
                rows.append(currentRow)
                currentRow=""
            rows.append(i)
        else:
            if currentRow!="":
                currentRow=currentRow+" "+i
            else:
                currentRow=i
            if random.random()<len(currentRow)/10 and len(currentRow)>3:#Use length in this stopping calculation
                rows.append(currentRow)
                currentRow=""

    if currentRow!="":
        if len(currentRow)<4:
            rows[-1]=rows[-1]+" "+currentRow
        else:
            rows.append(currentRow)

    longest=max(rows, key=len)
    maxSize=int((box[3]-box[1])/len(rows))#Height/rows

    validW=(box[2]-box[0])*0.9
    width=99999
    size=maxSize
    while width>validW:
        textFont = PIL.ImageFont.truetype(font, size)
        width,height = textFont.getsize(longest)
        size-=1

    textImg=PIL.Image.new("RGBA",(int(box[2]-box[0]),int(box[3]-box[1])),(0, 0, 0, 0))
    draw=PIL.ImageDraw.Draw(textImg)
    starty=0
    for i in range(len(rows)):
        width=0
        size=10
        while width<validW:
            textFont = PIL.ImageFont.truetype(font, size)
            width,height = textFont.getsize(rows[i])
            size+=1
            if size>=maxSize:
                width=99999
        width,height = textFont.getsize(rows[i])

        #Need to center THIS!
        draw.text(((box[2]-box[0]-width)/2,starty),rows[i],color,font=textFont,stroke_width=random.randint(0,2),stroke_fill=color)
        starty=starty+size

    width=int(box[2]-box[0])
    height=starty
    textImg=textImg.crop((0,0,width,starty))
    img.paste(textImg,(int(box[0]),int(box[1]+(box[3]-box[1]-height)/2)),textImg)
    return Idraw















#PLACES TEXT IN A TRADITIONAL WAY
def add_text(Idraw,text,subText,box,color,font):
    val=random.random()
    if val<0.25:
        text=text.upper()
    val=random.random()
    if val<0.25:
        subText=subText.upper()
    width=99999
    size=100
    while width>(box[2]-box[0])*0.9:
        textFont = PIL.ImageFont.truetype(font, size)
        width,height = textFont.getsize(text)
        size-=1
    #SubText
    subSize=int(size/2)
    subWidth=99999
    while subWidth>(box[2]-box[0])*0.8:
        subFont = PIL.ImageFont.truetype(font, subSize)
        subWidth,subHeight = subFont.getsize(subText)
        subSize-=1
    #Place in the middle
    Idraw.text((box[0]+(box[2]-box[0]-width)/2,box[1]+(box[3]-box[1]-height)/2),text,color,font=textFont)
    Idraw.text((box[0]+(box[2]-box[0]-subWidth)/2,box[1]+(box[3]-box[1]-height)/2+height*1.2),subText,color,font=subFont)














#COMPLETED######################################################################################################

#GETS A COLOUR FROM A LIST AND REMOVES IT - COMPLETED
def get_color(colors):
    if colors!=[]:
        color=random.choice(colors)
        colors.remove(color)
        return colors,color
    else:
        return colors,random.choice([(0,0,0),(255,255,255)])

#Gets 2 4-tuples as input - COMPLETED
def collide(box1,box2):
    if ((box1[0]>box2[0] and box1[0]<box2[2]) or (box1[2]>box2[0] and box1[2]<box2[2])) and ((box1[1]>box2[1] and box1[1]<box2[3]) or (box1[3]>box2[0] and box1[3]<box2[3])):
        return True
    return False

#SENDS AND RECIEVES A REQUEST FROM CHATGPT - COMPLETED
def chatGPT(message):
    messages=[{"role": "user", "content": message}]
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    text=chat.choices[0].message.content.split("\n")[0].strip("\"")
    return text

#PLACES A ROUND BOX WITH A BLUR EFFECT BELOW - COMPLETED
def blur_box(Idraw,img,w,h,imgw,imgh):
    # Create rectangle mask
    mask = PIL.Image.new('L', (imgw,imgh), 0)
    draw = PIL.ImageDraw.Draw(mask)
    draw.rounded_rectangle([ ((imgw-w)/2,(imgh-h)/2), ((imgw-w)/2+w,(imgh-h)/2+h) ], fill=255,radius=30)
    blurred = img.filter(PIL.ImageFilter.GaussianBlur(random.randint(5,20)))
    # Paste blurred region and save result
    img.paste(blurred, mask=mask)
    Idraw.rounded_rectangle([((imgw-w)/2,(imgh-h)/2), ((imgw-w)/2+w,(imgh-h)/2+h)],outline=(255,255,255,255),radius=30,width=2)
    textBox=((imgw-w)/2,(imgh-h)/2, (imgw-w)/2+w,(imgh-h)/2+h)
    return Idraw,img,textBox

#OVERLAYS THE IMAGE WITH A GIVEN COLOUR - COMPLETED
def color_fill(Idraw,img,col,imgw,imgh):
    overlay = PIL.Image.new('RGBA', (imgw,imgh), (col[0],col[1],col[2],200))
    # Alpha composite these two images together to obtain the desired result.
    img = PIL.Image.alpha_composite(img, overlay)
    return img

#CONVERTS HEX COLOUR TO RGB VALUE - COMPLETED
def hex_to_rgb(hex):
  rgb = []
  for i in (0, 2, 4):
    decimal = int(hex[i:i+2], 16)
    rgb.append(decimal)
  return tuple(rgb)



#IF RUNNING THIS PROGRAM DIRECTLY USE THIS
if __name__=="__main__":
    ex=AIImage(
        api_key="*******",
        text="Durable Computer and Accessories at Affordable Prices",
        subText="Get Assorted Computer and Accessories",
        logo="logo.png",
        description="A store selling computer and accessories",
        size=(1000,800),
        image="",
        colors="#29C2FF, #3B00FF, #000000, #FFFFFF",
        font="arial.ttf",
        target="",
        image_type="Instagram"
        )
    
    
    ex.generate_images(1)
    
    #Returns 16 random designs
    imgs=ex.get_designs(16)
    
    # Set the directory where you want to save the images
    output_directory = "output_images/"

    # Iterate through the list of images and save them to files
    for i, image in enumerate(imgs):
        # You can specify the format and filename here
        filename = f"image_{i+1}.png"  # Change the format and naming as needed
        image.save(output_directory + filename)

    print("Images saved successfully.")