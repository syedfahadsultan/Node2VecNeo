package embedding;


import org.deeplearning4j.graph.api.NoEdgeHandling;
import org.deeplearning4j.graph.api.Vertex;
import org.deeplearning4j.graph.models.deepwalk.DeepWalk;
import org.deeplearning4j.graph.models.deepwalk.GraphHuffman;
import org.deeplearning4j.graph.models.embeddings.InMemoryGraphLookupTable;
import org.neo4j.collection.primitive.PrimitiveIntIterator;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.graph.iterator.RandomWalkIterator;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.GraphFactory;
import org.neo4j.graphalgo.core.GraphLoader;
import org.neo4j.graphalgo.core.ProcedureConfiguration;
import org.neo4j.graphalgo.core.utils.Pools;
import org.neo4j.graphalgo.core.utils.ProgressTimer;
import org.neo4j.graphalgo.core.utils.TerminationFlag;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.core.write.Exporter;
import org.neo4j.graphalgo.results.PageRankScore;
import org.neo4j.graphdb.Direction;
import org.neo4j.kernel.api.KernelTransaction;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.logging.Log;
import org.neo4j.procedure.*;
import org.neo4j.values.storable.DeepWalkPropertyTranslator;
import org.neo4j.values.storable.LookupTablePropertyTranslator;

import java.util.*;
import java.util.stream.IntStream;
import java.lang.System;
import java.util.stream.Stream;

public class Node2VecProc {

    @Context
    public GraphDatabaseAPI api;

    @Context
    public Log log;

    @Context
    public KernelTransaction transaction;

    @Procedure(value = "embedding.dl4j.node2vec", mode = Mode.WRITE)
    @Description("CALL embedding.dl4j.node2vec(label:String, relationship:String, " +
            "{graph: 'heavy/cypher', vectorSize:128, windowSize:10, learningRate:0.01 concurrency:4, direction:'BOTH}) " +
            "YIELD nodes, iterations, loadMillis, computeMillis, writeMillis, dampingFactor, write, writeProperty" +
            " - calculates page rank and potentially writes back")
    public Stream<PageRankScore.Stats> node2Vec(
            @Name(value = "label", defaultValue = "") String label,
            @Name(value = "relationship", defaultValue = "") String relationship,
            @Name(value = "config", defaultValue = "{}") Map<String, Object> config) {

        long startTime = System.currentTimeMillis();
        ProcedureConfiguration configuration = ProcedureConfiguration.create(config);

        long elapsedTime = System.currentTimeMillis() - startTime;
        log.info("Deepwalk: Elapsed time [1]: "+elapsedTime);
        startTime = System.currentTimeMillis();

        AllocationTracker tracker = AllocationTracker.create();

        elapsedTime = System.currentTimeMillis() - startTime;
        log.info("Deepwalk: Elapsed time [2]: "+elapsedTime);
        startTime = System.currentTimeMillis();

        PageRankScore.Stats.Builder statsBuilder = new PageRankScore.Stats.Builder();

        elapsedTime = System.currentTimeMillis() - startTime;
        log.info("Deepwalk: Elapsed time [3]: "+elapsedTime);
        startTime = System.currentTimeMillis();

        final Graph graph = load(label, relationship, tracker, configuration.getGraphImpl(), statsBuilder, configuration);

        elapsedTime = System.currentTimeMillis() - startTime;
        log.info("Deepwalk: Elapsed time [4]: "+elapsedTime);
        startTime = System.currentTimeMillis();


        int nodeCount = Math.toIntExact(graph.nodeCount());
        if (nodeCount == 0) {
            graph.release();
            return Stream.empty();
        }

        elapsedTime = System.currentTimeMillis() - startTime;
        log.info("Deepwalk: Elapsed time [5]: "+elapsedTime);
        startTime = System.currentTimeMillis();


        org.deeplearning4j.graph.graph.Graph<Integer, Integer> iGraph = buildDl4jGraph(graph);
        Node2Vec<Integer, Integer> dw = runDeepWalk(iGraph, statsBuilder, configuration);

        elapsedTime = System.currentTimeMillis() - startTime;
        log.info("Deepwalk: Elapsed time [6]: "+elapsedTime);
        startTime = System.currentTimeMillis();

        if (configuration.isWriteFlag()) {
            final String writeProperty = configuration.getWriteProperty("deepWalk");

            elapsedTime = System.currentTimeMillis() - startTime;
            log.info("Deepwalk: Elapsed time [7]: "+elapsedTime);
            startTime = System.currentTimeMillis();


            statsBuilder.timeWrite(() -> Exporter.of(api, graph)
                    .withLog(log)
                    .parallel(Pools.DEFAULT, configuration.getConcurrency(), TerminationFlag.wrap(transaction))
                    .build()
                    .write(
                            writeProperty,
                            dw,
                            new DeepWalkPropertyTranslator()
                    )
            );

            elapsedTime = System.currentTimeMillis() - startTime;
            log.info("Deepwalk: Elapsed time [8]: "+elapsedTime);
            startTime = System.currentTimeMillis();

        }

        return Stream.of(statsBuilder.build());
    }


    @Procedure(name = "embedding.dl4j.node2vec.stream", mode = Mode.READ)
    @Description("CALL embedding.dl4j.node2vec.stream(label:String, relationship:String, {graph: 'heavy/cypher', walkLength:10, vectorSize:10, windowSize:2, learningRate:0.01 concurrency:4, direction:'BOTH'}) " +
            "YIELD nodeId, embedding - compute embeddings for each node")
    public Stream<DeepWalkResult> deepWalkStream(
            @Name(value = "label", defaultValue = "") String label,
            @Name(value = "relationship", defaultValue = "") String relationship,
            @Name(value = "config", defaultValue = "{}") Map<String, Object> config) {

        ProcedureConfiguration configuration = ProcedureConfiguration.create(config);
        AllocationTracker tracker = AllocationTracker.create();

        PageRankScore.Stats.Builder statsBuilder = new PageRankScore.Stats.Builder();
        final Graph graph = load(label, relationship, tracker, configuration.getGraphImpl(), statsBuilder, configuration);

        int nodeCount = Math.toIntExact(graph.nodeCount());
        if (nodeCount == 0) {
            graph.release();
            return Stream.empty();
        }

        org.deeplearning4j.graph.graph.Graph<Integer, Integer> iGraph = buildDl4jGraph(graph);
        Node2Vec<Integer, Integer> dw = runDeepWalk(iGraph, statsBuilder, configuration);

        return IntStream.range(0, dw.numVertices()).mapToObj(index ->
                new DeepWalkResult(graph.toOriginalNodeId(index), dw.getVertexVector(index).toDoubleVector()));
    }

    private org.deeplearning4j.graph.graph.Graph<Integer, Integer> buildDl4jGraph(Graph graph) {
        List<Vertex<Integer>> nodes = new ArrayList<>();

        PrimitiveIntIterator nodeIterator = graph.nodeIterator();
        while(nodeIterator.hasNext()) {
            int nodeId = nodeIterator.next();
            nodes.add(new Vertex<>(nodeId,nodeId));
        }

        boolean allowMultipleEdges = true;

        org.deeplearning4j.graph.graph.Graph<Integer, Integer> iGraph = new org.deeplearning4j.graph.graph.Graph<>(nodes, allowMultipleEdges);

        nodeIterator = graph.nodeIterator();
        while(nodeIterator.hasNext()) {
            int nodeId = nodeIterator.next();
            graph.forEachRelationship(nodeId, Direction.BOTH, (sourceNodeId, targetNodeId, relationId) -> {
                iGraph.addEdge(nodeId, targetNodeId, -1, false);
                return false;
            });
        }
        return iGraph;
    }

    private Node2Vec<Integer, Integer> runDeepWalk(org.deeplearning4j.graph.graph.Graph<Integer, Integer> iGraph,
                                                   PageRankScore.Stats.Builder statsBuilder, ProcedureConfiguration configuration) {
        
        long startTime = System.currentTimeMillis();

        long vectorSize = configuration.get("vectorSize", 128L);
        double learningRate = configuration.get("learningRate", 0.01);
        long  windowSize = configuration.get("windowSize", 10L);
        long walkLength = configuration.get("walkSize", 40L);
        long numberOfWalks = configuration.get("numberOfWalks", 80L);

        long elapsedTime = System.currentTimeMillis() - startTime;
        log.info("Deepwalk: Elapsed time [11]: "+elapsedTime);
        startTime = System.currentTimeMillis();

        Map<String, Number> params = new HashMap<>();
        params.put("vectorSize", vectorSize);
        params.put("learningRate", learningRate);
        params.put("windowSize", windowSize);
        params.put("walkLength", walkLength);
        params.put("numberOfWalks", numberOfWalks);

        elapsedTime = System.currentTimeMillis() - startTime;
        log.info("Deepwalk: Elapsed time [12]: "+elapsedTime);
        startTime = System.currentTimeMillis();

        log.info("Executing [Fahad] Node2Vec with params: %s", params);


        Node2Vec.Builder<Integer, Integer> builder = new Node2Vec.Builder<>();
        builder.vectorSize((int) vectorSize);
        builder.learningRate(learningRate);
        builder.windowSize((int) windowSize);
        Node2Vec<Integer, Integer> dw = builder.build();

        elapsedTime = System.currentTimeMillis() - startTime;
        log.info("Deepwalk: Elapsed time [13]: "+elapsedTime);
        startTime = System.currentTimeMillis();


        dw.initialize(iGraph, log);
        // GraphWalkIterator<Integer> iter = new RandomWalkIterator<>(iGraph,iGraph.numVertices());
        // statsBuilder.timeEval(()->dw.fit(iter));

        elapsedTime = System.currentTimeMillis() - startTime;
        log.info("Deepwalk: Elapsed time [14]: "+elapsedTime);
        startTime = System.currentTimeMillis();


        statsBuilder.timeEval(() -> dw.fit(new Node2VecIteratorProvider<>(
                iGraph, (int) walkLength, 1,
                NoEdgeHandling.SELF_LOOP_ON_DISCONNECTED, (int) numberOfWalks, log)));

        elapsedTime = System.currentTimeMillis() - startTime;
        log.info("Deepwalk: Elapsed time [15]: "+elapsedTime);
        startTime = System.currentTimeMillis();

        return dw;
    }


    private Graph load(
            String label,
            String relationship,
            AllocationTracker tracker,
            Class<? extends GraphFactory> graphFactory,
            PageRankScore.Stats.Builder statsBuilder, ProcedureConfiguration configuration) {
        GraphLoader graphLoader = new GraphLoader(api, Pools.DEFAULT)
                .init(log, label, relationship, configuration)
                .withAllocationTracker(tracker)
                .withDirection(configuration.getDirection(Direction.BOTH))
                .withoutNodeProperties()
                .withoutNodeWeights()
                .withoutRelationshipWeights();


        try (ProgressTimer timer = ProgressTimer.start()) {
            Graph graph = graphLoader.load(graphFactory);
            statsBuilder.withNodes(graph.nodeCount());
            return graph;
        }
    }

}

