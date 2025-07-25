/*
 * Copyright 2023-2025 the original author or authors.
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

package org.springframework.ai.embedding.observation;

import java.util.List;

import org.junit.jupiter.api.Test;

import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingOptionsBuilder;
import org.springframework.ai.embedding.EmbeddingRequest;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for {@link EmbeddingModelObservationContext}.
 *
 * @author Thomas Vitale
 */
class EmbeddingModelObservationContextTests {

	@Test
	void whenMandatoryRequestOptionsThenReturn() {
		var observationContext = EmbeddingModelObservationContext.builder()
			.embeddingRequest(generateEmbeddingRequest(EmbeddingOptionsBuilder.builder().model("supermodel").build()))
			.provider("superprovider")
			.build();

		assertThat(observationContext).isNotNull();
	}

	private EmbeddingRequest generateEmbeddingRequest(EmbeddingOptions embeddingOptions) {
		return new EmbeddingRequest(List.of(), embeddingOptions);
	}

}
