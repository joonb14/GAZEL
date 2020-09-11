package com.google.mlkit.vision.demo;

//https://www.geeksforgeeks.org/array-implementation-of-queue-simple/
// Java program to implement a queue using an array
public class Queue {
    public  int front, rear, capacity;
    public float[] queue;

    public Queue(int c){
        front = rear = 0;
        capacity = c;
        queue = new float[capacity];
    }
    public boolean isFull() {
        if (capacity == rear) {
            return true;
        }
        return false;
    }
    public boolean isEmpty() {
        if (front == rear) {
            return true;
        }
        return false;
    }

    // function to insert an element
    // at the rear of the queue
    public void queueEnqueue(float data){
        // check queue is full or not
        if (capacity == rear) {
            return;
        }
        // insert element at the rear
        else {
            queue[rear] = data;
            rear++;
        }
        return;
    }

    // function to delete an element
    // from the front of the queue
    public void queueDequeue(){
        // if queue is empty
        if (front == rear) {
            return;
        }
        // shift all the elements from index 2 till rear
        // to the right by one
        else {
            for (int i = 0; i < rear - 1; i++) {
                queue[i] = queue[i + 1];
            }
            // store 0 at rear indicating there's no element
            if (rear < capacity)
                queue[rear] = 0;
            // decrement rear
            rear--;
        }
        return;
    }
}