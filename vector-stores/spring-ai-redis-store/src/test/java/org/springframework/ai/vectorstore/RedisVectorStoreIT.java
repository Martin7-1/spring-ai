package org.springframework.ai.vectorstore;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.ai.autoconfigure.openai.OpenAiAutoConfiguration;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.util.FileSystemUtils;
import org.testcontainers.containers.DockerComposeContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import redis.clients.jedis.JedisPooled;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

@Testcontainers
public class RedisVectorStoreIT {

    @Container
    public static DockerComposeContainer redisContainer = new DockerComposeContainer(
            new File("src/test/resources/docker-compose.yml"))
            .withExposedService("standalone", 19530)
            .withExposedService("standalone", 9091,
                    Wait.forHttp("/healthz").forPort(9091)
                            .forStatusCode(200)
                            .forStatusCode(401));

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(TestApplication.class)
            .withPropertyValues("spring.ai.openai.apiKey=" + System.getenv("SPRING_AI_OPENAI_API_KEY"));

    List<Document> documents = List.of(
            new Document("Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!!",
                    Collections.singletonMap("meta1", "meta1")),
            new Document("Hello World Hello World Hello World Hello World Hello World Hello World Hello World"),
            new Document(
                    "Great Depression Great Depression Great Depression Great Depression Great Depression Great Depression",
                    Collections.singletonMap("meta2", "meta2")));

    @BeforeAll
    public static void beforeAll() throws IOException {
        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @AfterAll
    public static void afterAll() {
        FileSystemUtils.deleteRecursively(new File("src/test/resources/volumes"));
    }

    private void resetCollection(VectorStore vectorStore) {
        ((RedisVectorStore) vectorStore).dropIndex("test_vector_store");
        ((RedisVectorStore) vectorStore).createIndex("test_vector_store");
    }

    @Test
    public void addAndSearchTest() {

        contextRunner.withConfiguration(AutoConfigurations.of(OpenAiAutoConfiguration.class)).run(context -> {

            VectorStore vectorStore = context.getBean(VectorStore.class);

            resetCollection(vectorStore);

            vectorStore.add(documents);

            List<Document> results = vectorStore.similaritySearch("Great", 1);

            assertThat(results).hasSize(1);
            Document resultDoc = results.get(0);
            assertThat(resultDoc.getId()).isEqualTo(documents.get(2).getId());
            assertThat(resultDoc.getText()).isEqualTo(
                    "Great Depression Great Depression Great Depression Great Depression Great Depression Great Depression");
            assertThat(resultDoc.getMetadata()).isEqualTo(Collections.singletonMap("meta2", "meta2"));

            // Remove all documents from the store
            vectorStore.delete(documents.stream().map(Document::getId).toList());

            List<Document> results2 = vectorStore.similaritySearch("Hello", 1);
            assertThat(results2).hasSize(0);

        });
    }

    @Test
    public void documentUpdateTest() {

        contextRunner.withConfiguration(AutoConfigurations.of(OpenAiAutoConfiguration.class)).run(context -> {

            VectorStore vectorStore = context.getBean(VectorStore.class);

            resetCollection(vectorStore);

            Document document = new Document(UUID.randomUUID().toString(), "Spring AI rocks!!",
                    Collections.singletonMap("meta1", "meta1"));

            vectorStore.add(List.of(document));

            List<Document> results = vectorStore.similaritySearch("Spring", 5);

            assertThat(results).hasSize(1);
            Document resultDoc = results.get(0);
            assertThat(resultDoc.getId()).isEqualTo(document.getId());
            assertThat(resultDoc.getText()).isEqualTo("Spring AI rocks!!");
            assertThat(resultDoc.getMetadata()).isEqualTo(Collections.singletonMap("meta1", "meta1"));

            Document sameIdDocument = new Document(document.getId(),
                    "The World is Big and Salvation Lurks Around the Corner",
                    Collections.singletonMap("meta2", "meta2"));

            vectorStore.add(List.of(sameIdDocument));

            results = vectorStore.similaritySearch("FooBar", 5);

            assertThat(results).hasSize(1);
            resultDoc = results.get(0);
            assertThat(resultDoc.getId()).isEqualTo(document.getId());
            assertThat(resultDoc.getText()).isEqualTo("The World is Big and Salvation Lurks Around the Corner");
            assertThat(resultDoc.getMetadata()).isEqualTo(Collections.singletonMap("meta2", "meta2"));

            vectorStore.delete(List.of(document.getId()));

        });
    }

    @Test
    public void searchThresholdTest() {

        contextRunner.withConfiguration(AutoConfigurations.of(OpenAiAutoConfiguration.class)).run(context -> {

            VectorStore vectorStore = context.getBean(VectorStore.class);

            resetCollection(vectorStore);

            vectorStore.add(documents);

            assertThat(vectorStore.similaritySearch("Great", 5, 1.0)).hasSize(3);

            List<Document> results = vectorStore.similaritySearch("Great", 5, 0.43);

            assertThat(results).hasSize(1);
            Document resultDoc = results.get(0);
            assertThat(resultDoc.getId()).isEqualTo(documents.get(2).getId());
            assertThat(resultDoc.getText()).isEqualTo(
                    "Great Depression Great Depression Great Depression Great Depression Great Depression Great Depression");
            assertThat(resultDoc.getMetadata()).isEqualTo(Collections.singletonMap("meta2", "meta2"));

        });
    }

    @SpringBootConfiguration
    @EnableAutoConfiguration(exclude = {DataSourceAutoConfiguration.class})
    public static class TestApplication {

        @Bean
        public VectorStore vectorStore(JedisPooled jedisClient, EmbeddingClient embeddingClient) {
            return new RedisVectorStore(jedisClient, embeddingClient, "test_vector_store", "test_vector_store");
        }

        @Bean
        public JedisPooled jedisClient() {
            return new JedisPooled("localhost", 6379, "default", "password");
        }

    }
}
