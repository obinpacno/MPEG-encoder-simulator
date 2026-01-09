class Inter_Encoder{
  int blockSize;
  int searchRange;
  float Qp;
  float lambda;
  
  Inter_Encoder(int bS, int sR, float Qp, float lmb){
    this.blockSize = bS;
    this.searchRange = sR;
    this.Qp = Qp;
    this.lambda = lmb;
  }
  
  float[][] applyDCT(int[][] spatialBlock){
    int N = blockSize;
    float[][] dctCoeffs = new float[N][N];
    
    for(int u = 0; u < N; u++){
      for(int v = 0; v < N; v++){
        float sum = 0.0;
        
        for(int x = 0; x < N; x++){
          for(int y = 0; y < N; y++){
            color pixel = spatialBlock[y][x];
            float luminance = 0.299 * red(pixel) + 0.587 * green(pixel) + 0.114 * blue(pixel);
            
            float cos1 = cos(PI * u * (2*x + 1) / (2*N));
            float cos2 = cos(PI * v * (2*y + 1) / (2*N));
            sum += luminance * cos1 * cos2;
          }
        }
        float cu = (u == 0) ? 1.0/sqrt(2) : 1.0;
        float cv = (v == 0) ? 1.0/sqrt(2) : 1.0;
        dctCoeffs[u][v] = 0.25 * cu * cv * sum;
      }
    }
    return dctCoeffs;
  }
  
  float[][] applyInverseDCT(float[][] dctCoeffs){
    int N = blockSize;
    float[][] spatialBlock = new float[N][N];
    
    for(int x = 0; x < N; x++){
      for(int y = 0; y < N; y++){
        float sum = 0.0;
        
        for(int u = 0; u < N; u++){
          for(int v = 0; v < N; v++){
            float cu = (u == 0) ? 1.0/sqrt(2) : 1.0;
            float cv = (v == 0) ? 1.0/sqrt(2) : 1.0;
            
            float cos1 = cos(PI * u * (2*x + 1) / (2*N));
            float cos2 = cos(PI * v * (2*y + 1) / (2*N));
            sum += cu * cv * dctCoeffs[u][v] * cos1 * cos2;
          }
        }
        spatialBlock[y][x] = 0.25 * sum;
      }
    }
    return spatialBlock;
  }
  
  float[][] createQuantizationMatrix(int size){
    float[][] matrix = new float[size][size];
    for(int i = 0; i < size; i++){
      for(int j = 0; j < size; j++){
        matrix[i][j] = 1.0 + (i + j) * 0.3;
      }
    }
    return matrix;
  }
  
  float[][] applyQuantization(float[][] dctCoeffs, float Qp){
    int N = blockSize;
    float[][] quantized = new float[N][N];
    
    float[][] quantizationMatrix = createQuantizationMatrix(N);
    
    for(int u = 0; u < N; u++){
      for (int v = 0; v < N; v++){
        float qStep = quantizationMatrix[u][v] * (Qp / 10.0);
        if (abs(qStep) > 0.001)
          quantized[u][v] = round(dctCoeffs[u][v] / qStep);
        else
          quantized[u][v] = dctCoeffs[u][v];
      }
    }
    return quantized;
  }
  
  float[][] applyInverseQuantization(float[][] quantized, float Qp){
    int N = blockSize;
    float[][] dequantized = new float[N][N];
    
    float[][] quantizationMatrix = createQuantizationMatrix(N);
    
    for(int u = 0; u < N; u++){
      for(int v = 0; v < N; v++){
        float qStep = quantizationMatrix[u][v] * (Qp / 10.0);
        dequantized[u][v] = quantized[u][v] * qStep;
      }
    }
    return dequantized;
  }

  int SAD(int[][] block1, int[][] block2){
    int sad = 0;
    float red, green, blue;
    
    for(int j = 0; j < blockSize; j++){
      for(int i = 0; i < blockSize; i++){
        color c1 = block1[j][i];
        color c2 = block2[j][i];
        
        red = abs((int)red(c1) - (int)red(c2));
        green = abs((int)green(c1) - (int)green(c2));
        blue = abs((int)blue(c1) - (int)blue(c2));
        
        float diff = red + green + blue;
        sad += diff;
      }
    }
    
    sad /= 3 * blockSize * blockSize;
    
    return sad;
  }
  
  int[][] extractBlock(PImage img, int x, int y, int size){
    int[][] block = new int[size][size];
    for (int j = 0; j < size; j++)
      for (int i = 0; i < size; i++)
        block[j][i] = img.get(x + i, y + j);
    return block;
  }
  
  int[][] calculateResidual(int[][] original, int[][] predicted){
    int[][] residual = new int[blockSize][blockSize];
    for(int j = 0; j < blockSize; j++){
      for(int i = 0; i < blockSize; i++){
        residual[j][i] = original[j][i] - predicted[j][i];
      }
    }
    return residual;
  }
  
  float calculateDistortion(int[][] original, int[][] predicted, float qp, int[][] residual){
    float[][] dctResidual = applyDCT(residual);
    float[][] quantized = applyQuantization(dctResidual, qp);
    float[][] dequantized = applyInverseQuantization(quantized, qp);
    float[][] reconstructedResidual = applyInverseDCT(dequantized);
    
    float mse = 0;
    int count = 0;
    for(int i = 0; i < blockSize; i++){
      for(int j = 0; j < blockSize; j++){
        color origPixel = original[i][j];
        float origLuma = 0.299 * red(origPixel) + 0.587 * green(origPixel) + 0.114 * blue(origPixel);
        
        color predPixel = predicted[i][j];
        float predLuma = 0.299 * red(predPixel) + 0.587 * green(predPixel) + 0.114 * blue(predPixel);
        
        float reconstructedLuma = predLuma + reconstructedResidual[i][j];
        float error = origLuma - reconstructedLuma;
        mse += error * error;
        count++;
      }
    }
    mse /= count;
    
    return mse * 10000;
  }
  
  NeighborMVs getNeighborMotionVectors(int startX, int startY){
    MotionVector above = new MotionVector(0, -2);
     
    MotionVector left = new MotionVector(-1, 0);
    
    MotionVector aboveRight = new MotionVector(1, -1);
    
    if (startY < blockSize) above = new MotionVector(0, 0);
    if (startX < blockSize) left = new MotionVector(0, 0);
    if (startX >= width/2 - blockSize) aboveRight = new MotionVector(0, 0);
    
    return new NeighborMVs(above, left, aboveRight);
  }
  
  MotionVector predictMVFromNeighbors(NeighborMVs neighbors){
    int median_dx = median(neighbors.above.dx, neighbors.left.dx, neighbors.aboveRight.dx);
    int median_dy = median(neighbors.above.dy, neighbors.left.dy, neighbors.aboveRight.dy);
    return new MotionVector(median_dx, median_dy);
  }
  
  int median(int a, int b, int c){
    if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
    if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
    return c;
  }
  
  float estimateExpGolombBits(int value){
    int abs_val = abs(value);
    if (abs_val == 0) return 1.0;
    
    int k = 0;
    while(abs_val >= (1 << k)){
      k++;
    }
    return 2.0 * k + 1.0;
  }
  
  float estimateBitsMV(int dx, int dy, int startX, int startY){
    NeighborMVs neighbors = getNeighborMotionVectors(startX, startY);
    MotionVector predicted_mv = predictMVFromNeighbors(neighbors);
    
    int mvd_x = dx - predicted_mv.dx;
    int mvd_y = dy - predicted_mv.dy;
    
    float bits = estimateExpGolombBits(mvd_x) + estimateExpGolombBits(mvd_y);
    
    return bits;
  }
  
  ArrayList<Float> getZigzagCoefficients(float[][] block){
    ArrayList<Float> zigzag = new ArrayList<Float>();
    int n = block.length;
    
    if(n == 8){
      // Zig-zag per 8x8 (JPEG standard)
      int[] zigzagOrder = {
         0,  1,  5,  6, 14, 15, 27, 28,
         2,  4,  7, 13, 16, 26, 29, 42,
         3,  8, 12, 17, 25, 30, 41, 43,
         9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
      };
      
      for(int i = 0; i < zigzagOrder.length; i++){
        int row = zigzagOrder[i] / n;
        int col = zigzagOrder[i] % n;
        zigzag.add(block[row][col]);
      }
    } 
    else
      generateAdaptiveZigzag(block, zigzag); // Zig-zag generico per qualsiasi dimensione
    
    return zigzag;
  }
  
  void generateAdaptiveZigzag(float[][] block, ArrayList<Float> zigzag){
    int n = block.length;
    int i = 0, j = 0;
    boolean goingUp = true;
    
    zigzag.add(block[i][j]);
    
    while(i != n - 1 || j != n - 1){
      if(goingUp){
        if(i > 0 && j < n - 1){
          i--;
          j++;
        } 
        else{
          goingUp = false;
          if(j < n - 1)
            j++;
          else
            i++;
        }
      } 
      else{
        if(j > 0 && i < n - 1){
          i++;
          j--;
        } 
        else{
          goingUp = true;
          if(i < n - 1)
            i++;
          else
            j++;
        }
      }
      zigzag.add(block[i][j]);
    }
  }
  
  int countNonZeroCoeffs(ArrayList<Float> coefficients){
    int count = 0;
    for(float coeff : coefficients){
      if(abs(coeff) > 0.001)
        count++;
    }
    return count;
  }
  
  float estimateCoefficientBits(float coeff){
    int absVal = (int)abs(coeff);
    
    if(absVal == 0) 
      return 0.0;

    if(absVal == 1)
      return 2.0;
      
    else
      return 1.0 + estimateExpGolombBits(absVal - 1);
  }
  
  int countTrailingOnes(ArrayList<Float> coefficients){
    int trailingOnes = 0;
    
    for(int i = coefficients.size() - 1; i >= 0; i--){
      float coeff = coefficients.get(i);
      if(abs(coeff) > 0.001){
        if(abs(abs(coeff) - 1.0) < 0.001)
          trailingOnes++;
        else
          break;
      }
    }
    return trailingOnes;
  }
  
  float estimateBitsResidual(int[][] residual, float qp) {
    float[][] dctCoeffs = applyDCT(residual);
    
    float[][] quantized = applyQuantization(dctCoeffs, qp);
    
    return estimateBitsFromQuantizedCoeffs(quantized, qp);
  }
  
  float estimateBitsFromQuantizedCoeffs(float[][] quantized, float qp) {
    float totalBits = 0.0;
    int nonZeroCoeffs = 0;
    
    ArrayList<Float> zigzagCoeffs = getZigzagCoefficients(quantized);
    
    int numNonZero = countNonZeroCoeffs(zigzagCoeffs);
    if(numNonZero > 0)
      totalBits += estimateExpGolombBits(numNonZero);

    int run = 0;
    for(int i = 0; i < zigzagCoeffs.size(); i++){
      float coeff = zigzagCoeffs.get(i);
      
      if(abs(coeff) > 0.001){
        nonZeroCoeffs++;
        
        if(run > 0){
          totalBits += estimateExpGolombBits(run);
        }
        
        totalBits += estimateCoefficientBits(coeff);
        
        run = 0;
      } 
      else
        run++;
    }

    if(nonZeroCoeffs > 0){
      int trailingOnes = countTrailingOnes(zigzagCoeffs);
      totalBits += min(trailingOnes, 3) * 1.5;
      totalBits += log(nonZeroCoeffs + 1) * 0.5;
    }

    float qpFactor = 1.0 + (qp / 50.0);
    totalBits /= qpFactor;
    
    return max(2.0, totalBits);
  }
  
  float MotionEstimation(int startX, int startY, PImage currentFrame, PImage referenceFrame){
    int[][] currentBlock = extractBlock(currentFrame, startX, startY, blockSize);
    
    int bestDx = 0, bestDy = 0;
    int bestSad = Integer.MAX_VALUE;
    ArrayList<PVector> candidates = new ArrayList<PVector>();
    
    for (int dy = -searchRange; dy <= searchRange; dy += 2){
      for (int dx = -searchRange; dx <= searchRange; dx += 2){
        int candidateX = startX + dx;
        int candidateY = startY + dy;
        
        if (candidateX >= 0 && candidateX <= width/2 - blockSize && candidateY >= 0 && candidateY <= height - blockSize) {
          int[][] candidateBlock = extractBlock(referenceFrame, candidateX, candidateY, blockSize);
          
          int sad = SAD(currentBlock, candidateBlock);
          
          candidates.add(new PVector(dx, dy, sad));
         
          if (sad < bestSad){
            bestSad = sad;
            bestDx = dx;
            bestDy = dy;
          }
        }
      }
    }
        
    int[][] predictedBlock = extractBlock(referenceFrame, startX + bestDx, startY + bestDy, blockSize);
    
    int[][] residualBlock = calculateResidual(currentBlock, predictedBlock);
    
    float D_inter = calculateDistortion(currentBlock, predictedBlock, Qp, residualBlock);
    
    float bits_mv = estimateBitsMV(bestDx, bestDy, startX, startY);
    float bits_residual = estimateBitsResidual(residualBlock, Qp);
    float R_inter = bits_mv + bits_residual;
    
    float J_inter = D_inter + lambda * R_inter;
    
    return J_inter;
  }
}
