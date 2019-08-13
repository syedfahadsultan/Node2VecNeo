package org.neo4j.values.storable;

// import org.deeplearning4j.graph.models.deepwalk.DeepWalk;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.neo4j.graphalgo.core.write.PropertyTranslator;
import embedding.Node2Vec;

public class DeepWalkPropertyTranslator implements PropertyTranslator<Node2Vec<Integer, Integer>> {
    @Override
    public Value toProperty(int propertyId, Node2Vec<Integer, Integer> data, long nodeId) {

        INDArray row = data.getVertexVector((int) nodeId);

        double[] rowAsDouble = new double[row.size(1)];
        for (int columnIndex = 0; columnIndex < row.size(1); columnIndex++) {
            rowAsDouble[columnIndex] = row.getDouble(columnIndex);
        }

        return new DoubleArray(rowAsDouble);
    }
}