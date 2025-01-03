/*
 * Copyright 2023-2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.bedrock.cohere.api;

import java.time.Duration;
import java.util.List;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import software.amazon.awssdk.auth.credentials.EnvironmentVariableCredentialsProvider;
import software.amazon.awssdk.regions.Region;

import org.springframework.ai.bedrock.RequiresAwsCredentials;
import org.springframework.ai.bedrock.cohere.api.CohereChatBedrockApi.CohereChatModel;
import org.springframework.ai.bedrock.cohere.api.CohereChatBedrockApi.CohereChatRequest;
import org.springframework.ai.bedrock.cohere.api.CohereChatBedrockApi.CohereChatRequest.Truncate;
import org.springframework.ai.bedrock.cohere.api.CohereChatBedrockApi.CohereChatResponse;
import org.springframework.ai.bedrock.cohere.api.CohereChatBedrockApi.CohereChatResponse.Generation.FinishReason;
import org.springframework.ai.model.ModelOptionsUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * @author Christian Tzolov
 */
@RequiresAwsCredentials
public class CohereChatBedrockApiIT {

	private CohereChatBedrockApi cohereChatApi = new CohereChatBedrockApi(CohereChatModel.COHERE_COMMAND_V14.id(),
			EnvironmentVariableCredentialsProvider.create(), Region.US_EAST_1.id(), ModelOptionsUtils.OBJECT_MAPPER,
			Duration.ofMinutes(2));

	@Test
	public void requestBuilder() {

		CohereChatRequest request1 = new CohereChatRequest(
				"What is the capital of Bulgaria and what is the size? What it the national anthem?", 0.5, 0.9, 15, 40,
				List.of("END"), CohereChatRequest.ReturnLikelihoods.ALL, false, 1, null, Truncate.NONE);

		var request2 = CohereChatRequest
			.builder("What is the capital of Bulgaria and what is the size? What it the national anthem?")
			.temperature(0.5)
			.topP(0.9)
			.topK(15)
			.maxTokens(40)
			.stopSequences(List.of("END"))
			.returnLikelihoods(CohereChatRequest.ReturnLikelihoods.ALL)
			.stream(false)
			.numGenerations(1)
			.logitBias(null)
			.truncate(Truncate.NONE)
			.build();

		assertThat(request1).isEqualTo(request2);
	}

	@Test
	@Disabled("Due to model version has reached the end of its life")
	public void chatCompletion() {

		var request = CohereChatRequest
			.builder("What is the capital of Bulgaria and what is the size? What it the national anthem?")
			.stream(false)
			.temperature(0.5)
			.topP(0.8)
			.topK(15)
			.maxTokens(100)
			.stopSequences(List.of("END"))
			.returnLikelihoods(CohereChatRequest.ReturnLikelihoods.ALL)
			.numGenerations(3)
			.logitBias(null)
			.truncate(Truncate.NONE)
			.build();

		CohereChatResponse response = this.cohereChatApi.chatCompletion(request);

		assertThat(response).isNotNull();
		assertThat(response.prompt()).isEqualTo(request.prompt());
		assertThat(response.generations()).hasSize(request.numGenerations());
		assertThat(response.generations().get(0).text()).isNotEmpty();
	}

	@Disabled("Due to model version has reached the end of its life")
	@Test
	public void chatCompletionStream() {

		var request = CohereChatRequest
			.builder("What is the capital of Bulgaria and what is the size? What it the national anthem?")
			.stream(true)
			.temperature(0.5)
			.topP(0.8)
			.topK(15)
			.maxTokens(100)
			.stopSequences(List.of("END"))
			.returnLikelihoods(CohereChatRequest.ReturnLikelihoods.ALL)
			.numGenerations(3)
			.logitBias(null)
			.truncate(Truncate.NONE)
			.build();

		Flux<CohereChatResponse.Generation> responseStream = this.cohereChatApi.chatCompletionStream(request);
		List<CohereChatResponse.Generation> responses = responseStream.collectList().block();

		assertThat(responses).isNotNull();
		assertThat(responses).hasSizeGreaterThan(10);
		assertThat(responses.get(0).text()).isNotEmpty();

		CohereChatResponse.Generation lastResponse = responses.get(responses.size() - 1);
		assertThat(lastResponse.text()).isNull();
		assertThat(lastResponse.isFinished()).isTrue();
		assertThat(lastResponse.finishReason()).isEqualTo(FinishReason.MAX_TOKENS);
		assertThat(lastResponse.amazonBedrockInvocationMetrics()).isNotNull();
	}

	@Test
	public void testStreamConfigurations() {
		var streamRequest = CohereChatRequest
			.builder("What is the capital of Bulgaria and what is the size? What it the national anthem?")
			.stream(true)
			.build();

		assertThatThrownBy(() -> this.cohereChatApi.chatCompletion(streamRequest))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("The request must be configured to return the complete response!");

		var notStreamRequest = CohereChatRequest
			.builder("What is the capital of Bulgaria and what is the size? What it the national anthem?")
			.stream(false)
			.build();

		assertThatThrownBy(() -> this.cohereChatApi.chatCompletionStream(notStreamRequest))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("The request must be configured to stream the response!");

	}

}
