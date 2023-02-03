package com.github.jelmerk.simjoin.examples;

import java.util.Arrays;

import com.github.jelmerk.knn.Item;

public class Vector implements Item<Integer, double[]> {

    private static final long serialVersionUID = 1L;

    private final int id;
    private final int foreingKey;
    private final double[] vector;

    public Vector(int id, int foreingKey, double[] vector) {
        this.id = id;
        this.foreingKey = foreingKey;
        this.vector = vector;
    }

    @Override
    public Integer id() {
        return id;
    }
    
    public int foreingKey() {
        return foreingKey;
    }

    @Override
    public double[] vector() {
        return vector;
    }

    @Override
    public int dimensions() {
        return vector.length;
    }

    @Override
    public String toString() {
        return "Word{" +
                "id='" + id + '\'' +
                "foreignKey='" + foreingKey + '\'' +
                ", vector=" + Arrays.toString(vector) +
                '}';
    }
}