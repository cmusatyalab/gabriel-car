package edu.cmu.cs.gabrielclient;

import java.util.List;

public class VizObj {
    public String class_name;
    public List<Float> dimensions;
    public double confidence;
    public List<Float> norm;
    public boolean good_frame;

    VizObj() {
        // empty constructor
    }
}
