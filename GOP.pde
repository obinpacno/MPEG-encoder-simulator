class GOP{
  ArrayList<PImage> Frames;
  ArrayList<Character> Frames_Type;
  ArrayList<Integer> Frames_Color;
  int gop_size;
  int max_b_frames;
  int blockSize;
  float scene_change_threshold;
  float intra_threshold;
  
  Inter_Encoder inter;
  Intra_Encoder intra;
  
  GOP(ArrayList<PImage> f, int g, int b, int bS, int sR, float Qp, float lmb, float st, float it){
    this.Frames = f;
    this.gop_size = g;
    this.max_b_frames = b;
    this.blockSize = bS;
    this.scene_change_threshold = st;
    this.intra_threshold = it;
    Frames_Type = new ArrayList<Character>();
    Frames_Color = new ArrayList<Integer>();
    inter = new Inter_Encoder(bS, sR, Qp, lmb);
    intra = new Intra_Encoder(bS, Qp, lmb);
  }
  
  boolean checkThreshold(PImage frame, PImage key_frame){
    float j_intra, j_inter;
    int intraCounter = 0;
    int totalBlocks = (frame.width * frame.height) / (blockSize * blockSize);
    
    for(int i = 0; i < frame.width; i+=blockSize){
      for(int j = 0; j < frame.height; j+=blockSize){
        j_intra = intra.IntraCoding(i, j, frame);
        j_inter = inter.MotionEstimation(i, j, frame, key_frame);
        if(j_intra < j_inter)
          intraCounter++;
      }
    }
    
    float intra_percentage = (float(intraCounter) / float(totalBlocks)) * 100.0;
    
    if(intra_percentage >= intra_threshold)
      return true;
    return false;
  }
  
  boolean checkSceneChange(PImage frame1, PImage frame2){
    if(SAD(frame1, frame2) > this.scene_change_threshold)
      return true;
    return false;
  }
  
  float SAD(PImage frame1, PImage frame2){
    float sad = 0;
    float red, blue, green;
    for(int i = 0; i < frame1.width; i++){
      for(int j = 0; j < frame1.height; j++){
        red = abs(red(frame1.get(i,j)) - red(frame2.get(i,j)));
        green = abs(green(frame1.get(i,j)) - green(frame2.get(i,j)));
        blue = abs(blue(frame1.get(i,j)) - blue(frame2.get(i,j)));
        sad += red + green + blue;
      }
    }
    sad = sad/(3 * frame1.width * frame1.height);
    return sad;
  }
  
  void frames_classification(){
    int counter_i = 0;
    int counter_b = 0;
    boolean i_frame = true;
    PImage last_key_frame = Frames.get(0);
    int i = 0;
    
    for(PImage f: Frames){
      i++;
      if(i_frame == true){
        i_frame = false;
        counter_i++;
        last_key_frame = f;
        Frames_Type.add('I');
        Frames_Color.add(color(0,255,255));
        f.save("results/frame" + i + "_I_b.png");
        continue;
      }
      
      if(checkSceneChange(f, last_key_frame) == true){
        i_frame = false;
        counter_i = 1;
        counter_b = 0;
        last_key_frame = f;
        Frames_Type.add('I');
        Frames_Color.add(color(255,255,0));
        f.save("results/frame" + i + "_I_y.png");
        continue;
      }
      
      if(checkThreshold(f, last_key_frame) == true){
        i_frame = false;
        counter_i = 1;
        counter_b = 0;
        last_key_frame = f;
        Frames_Type.add('I');
        Frames_Color.add(color(255,0,0));
        f.save("results/frame" + i + "_I_r.png");
        continue;
      }
      
      if(counter_b < this.max_b_frames){
        Frames_Type.add('B');
        counter_b++;
        f.save("results/frame" + i + "_B.png");
      }
      else{
        Frames_Type.add('P');
        counter_b = 0;
        f.save("results/frame" + i + "_P.png");
      }
      
      if(counter_i >= this.gop_size - 1){
        counter_i = 0;
        counter_b = 0;
        i_frame = true;
      }
      else
        counter_i++;
      Frames_Color.add(color(240));
      }
  }
  
  ArrayList<Character> return_classification(){
    return this.Frames_Type;
  }
  
  ArrayList<Integer> return_color(){
    return this.Frames_Color;
  }
}
