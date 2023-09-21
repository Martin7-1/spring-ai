package org.springframework.ai.vectorstore;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.context.SmartLifecycle;
import redis.clients.jedis.JedisPooled;
import redis.clients.jedis.Pipeline;
import redis.clients.jedis.search.*;
import redis.clients.jedis.search.schemafields.SchemaField;
import redis.clients.jedis.search.schemafields.TextField;
import redis.clients.jedis.search.schemafields.VectorField;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

/**
 * Redis Integration as Vector Store
 */
public class RedisVectorStore implements VectorStore, SmartLifecycle {

    public static final int OPENAI_EMBEDDING_DIMENSION_SIZE = 1536;

    public static final String JSON_PATH_PREFIX = "$.";

    public static final String DOC_ID_FIELD_NAME = "doc_id";

    public static final String CONTENT_FIELD_NAME = "content";

    public static final String METADATA_FIELD_NAME = "metadata";

    public static final String EMBEDDING_FIELD_NAME = "embedding";

    public static final String DISTANCE_FIELD_NAME = "distance";

    public static final List<String> SEARCH_OUTPUT_FIELDS = Arrays.asList(DOC_ID_FIELD_NAME, CONTENT_FIELD_NAME,
            METADATA_FIELD_NAME);
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private final JedisPooled jedisClient;
    private final EmbeddingClient embeddingClient;
    private final int dimensions;
    private final String indexName;
    private final String prefix;
    private final RedisMetricType metricType;
    private final VectorField.VectorAlgorithm algorithm;

    public RedisVectorStore(JedisPooled jedisClient, EmbeddingClient embeddingClient, String prefix, String indexName) {
        this(jedisClient, embeddingClient, OPENAI_EMBEDDING_DIMENSION_SIZE, prefix, indexName,
                RedisMetricType.COSINE, VectorField.VectorAlgorithm.HNSW);
    }

    public RedisVectorStore(JedisPooled jedisClient, EmbeddingClient embeddingClient, int dimensions, String prefix,
                            String indexName, RedisMetricType metricType, VectorField.VectorAlgorithm algorithm) {
        this.jedisClient = jedisClient;
        this.embeddingClient = embeddingClient;
        this.dimensions = dimensions;
        this.indexName = indexName;
        this.prefix = prefix;
        this.metricType = metricType;
        this.algorithm = algorithm;
    }

    @Override
    public void add(List<Document> documents) {
        Pipeline pipeline = jedisClient.pipelined();
        documents.forEach(document -> pipeline.jsonSetWithEscape(prefix + ":" + document.getId(), document));
        pipeline.close();
    }

    @Override
    public Optional<Boolean> delete(List<String> idList) {
        Pipeline pipeline = jedisClient.pipelined();
        idList.forEach(pipeline::del);
        List<Object> results = pipeline.syncAndReturnAll();

        return Optional.of(results.stream().allMatch("OK"::equals));
    }

    @Override
    public List<Document> similaritySearch(String query) {
        return similaritySearch(query, 4);
    }

    @Override
    public List<Document> similaritySearch(String query, int k) {
        return similaritySearch(query, k, 1.0D);
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<Document> similaritySearch(String query, int k, double threshold) {
        List<Double> embedding = embeddingClient.embed(query);
        String queryTemplate = "*=>[ KNN %d @%s $BLOB AS %s ]";
        Query queryScript = new Query(String.format(queryTemplate, k, EMBEDDING_FIELD_NAME, DISTANCE_FIELD_NAME))
                .addParam("BLOB", toByteArray(embedding.stream().mapToDouble(Double::doubleValue).toArray()))
                .returnFields(SEARCH_OUTPUT_FIELDS.toArray(new String[0]))
                .setSortBy(DISTANCE_FIELD_NAME, true)
                .dialect(2);

        SearchResult result = jedisClient.ftSearch(indexName, queryScript);
        List<redis.clients.jedis.search.Document> documents = result.getDocuments();
        return documents.stream()
                .filter(document -> Double.parseDouble(document.getString(DISTANCE_FIELD_NAME)) < threshold)
                .map(document -> {
                    Map<String, Object> metadata;
                    try {
                        metadata = OBJECT_MAPPER.readValue(document.getString(METADATA_FIELD_NAME), Map.class);
                    } catch (JsonProcessingException e) {
                        throw new RuntimeException(e);
                    }

                    return new Document(document.getString(DOC_ID_FIELD_NAME), document.getString(CONTENT_FIELD_NAME), metadata);
                }).toList();
    }

    @Override
    public void start() {
        if (!hasIndex(indexName)) {
            createIndex(indexName);
        }
    }

    @Override
    public void stop() {
        if (hasIndex(indexName)) {
            dropIndex(indexName);
        }
    }

    @Override
    public boolean isRunning() {
        return hasIndex(indexName);
    }

    public void createIndex(String indexName) {
        IndexDefinition indexDefinition = new IndexDefinition(IndexDefinition.Type.JSON);
        indexDefinition.setPrefixes(prefix);
        String res = jedisClient.ftCreate(indexName, FTCreateParams.createParams()
                .on(IndexDataType.JSON)
                .addPrefix(prefix), createSchemaFields());
        if (!"OK".equals(res)) {
            throw new RuntimeException("create index error, msg=" + res);
        }
    }

    public void dropIndex(String indexName) {
        String res = jedisClient.ftDropIndex(indexName);
        if (!"OK".equals(res)) {
            throw new RuntimeException("create index error, msg=" + res);
        }
    }

    private SchemaField[] createSchemaFields() {
        Map<String, Object> vectorAttrs = new HashMap<>();
        vectorAttrs.put("DIM", dimensions);
        vectorAttrs.put("DISTANCE_METRIC", metricType.name());
        vectorAttrs.put("TYPE", "FLOAT64");
        vectorAttrs.put("INITIAL_CAP", 5);
        List<SchemaField> fields = new ArrayList<>();
        fields.add(TextField.of(JSON_PATH_PREFIX + DOC_ID_FIELD_NAME).as(DOC_ID_FIELD_NAME).weight(1.0));
        fields.add(TextField.of(JSON_PATH_PREFIX + CONTENT_FIELD_NAME).as(CONTENT_FIELD_NAME).weight(1.0));
        fields.add(VectorField.builder()
                .fieldName(JSON_PATH_PREFIX + EMBEDDING_FIELD_NAME)
                .algorithm(algorithm)
                .attributes(vectorAttrs)
                .as(EMBEDDING_FIELD_NAME)
                .build());
        fields.add(TextField.of(JSON_PATH_PREFIX + METADATA_FIELD_NAME).as(METADATA_FIELD_NAME).weight(1.0));

        return fields.toArray(new SchemaField[0]);
    }

    private boolean hasIndex(String indexName) {
        Set<String> indexList = jedisClient.ftList();
        return indexList.contains(indexName);
    }

    private byte[] toByteArray(double[] input) {
        byte[] bytes = new byte[Double.BYTES * input.length];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().put(input);
        return bytes;
    }

    public enum RedisMetricType {

        IP,
        L2,
        COSINE
    }
}
