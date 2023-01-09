# Neural Network

## Introduction
開始執行程式後，請輸入隱藏層節點數，輸入後即可看到神經網路學習 XOR 的過程，顯示結果，結果顯示訓練後的模型較符合真實情況

## Compile

gcc -c src/main.c -o build/main.o

gcc -c src/shuff.c -o build/shuff.o

gcc build/main.o build/shuff.o -o bin/main -lm

## Run
./bin/main

## Reference
1.參考程式碼 [https://github.com/niconielsen32/NeuralNetworks/blob/main/neuralNetC.c](https://github.com/niconielsen32/NeuralNetworks/blob/main/neuralNetC.c).

2.void指標轉型
[https://medium.com/@racktar7743/c%E8%AA%9E%E8%A8%80-%E6%8C%87%E6%A8%99%E6%95%99%E5%AD%B8-%E4%BA%94-1-void-pointer-c1cb976712a3](https://medium.com/@racktar7743/c%E8%AA%9E%E8%A8%80-%E6%8C%87%E6%A8%99%E6%95%99%E5%AD%B8-%E4%BA%94-1-void-pointer-c1cb976712a3).

3.動態記憶體配置 [https://blog.gtwang.org/programming/c-memory-functions-malloc-free/](https://blog.gtwang.org/programming/c-memory-functions-malloc-free/).