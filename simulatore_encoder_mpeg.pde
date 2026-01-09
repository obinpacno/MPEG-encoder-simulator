int gop_size;
int max_b_frames;
int blockSize;
int searchRange;
float Qp;
float lambda;
float scene_change_threshold;
float intra_threshold;

PImage img;
char c;
int index;

ArrayList<PImage> Frames = new ArrayList<PImage>();
ArrayList<Character> Frames_Type = new ArrayList<Character>();
ArrayList<Integer> Frames_Color = new ArrayList<Integer>();

GOP gop;

void setup() {
  size(500,240);
  textSize(50);
  textAlign(CENTER,CENTER);
  
  gop_size = 50;
  max_b_frames = 3;
  blockSize = 8;
  searchRange = 8;
  Qp = 26.0;
  lambda = 0.85;
  
  scene_change_threshold = 35.0;
  intra_threshold = 70.0;
  
  
  index = 0;
  //img = loadImage("video/0001.png");
  for(int i = 1020; i <= 1240; i++){
    if(i < 10){
      img = loadImage("video/000"+i+".png");
    }
    if(i >= 10 && i < 100){
      img = loadImage("video/00"+i+".png");
    }
    if(i >= 100 && i <1000){
      img = loadImage("video/0"+i+".png");
    }
    if(i >= 1000){
      img = loadImage("video/"+i+".png");
    }
    img.resize(426,240);
    Frames.add(img);
  }
  
  gop = new GOP(Frames, gop_size, max_b_frames, blockSize, searchRange, Qp, lambda, scene_change_threshold, intra_threshold);
  gop.frames_classification();
  Frames_Type = gop.return_classification();
  Frames_Color = gop.return_color();
  println(index+1);
}

void draw() {
  background(0);
  img = Frames.get(index);
  image(img,0,0);
  c = Frames_Type.get(index);
  //fill(255);
  //ellipse(463,height/2,50,50);
  fill(Frames_Color.get(index));
  text(c,463,height/2);
}

void keyPressed() {
  if(key == CODED) {
    if(keyCode == UP && index < 1240-720)
      index++;
    if(keyCode == DOWN && index > 0)
      index--;
    println(index+1);
    }
  if(key == ENTER)
   save("frame.png");
}
