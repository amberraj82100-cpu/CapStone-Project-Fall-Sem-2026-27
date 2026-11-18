package com.screening.service;

import com.screening.model.AssessmentSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Talks to the Python ML service (ml/service/app.py). Every call fails soft: if the service is down,
 * the assessment still completes with the rule-based report.
 */
@Service
public class MlServiceClient {

    private static final Logger log = LoggerFactory.getLogger(MlServiceClient.class);
    private final RestClient client;

    public MlServiceClient(RestClient.Builder builder,
                           @Value("${ml.service.url:http://localhost:8000}") String baseUrl) {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(2_000);
        factory.setReadTimeout(60_000); // the first call loads the CNN
        this.client = builder.baseUrl(baseUrl).requestFactory(factory).build();
    }

    /** Score a finished session; returns null when the ML service is unavailable. */
    public Map<String, Object> score(AssessmentSession session) {
        try {
            return client.post().uri("/api/score")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(session)
                    .retrieve()
                    .body(new ParameterizedTypeReference<Map<String, Object>>() {});
        } catch (Exception e) {
            log.warn("ML service scoring unavailable, using rule-based report only: {}", e.getMessage());
            return null;
        }
    }

    /** Ask the ML service to retrain if enough new labels have arrived. Runs in the background. */
    public void retrainIfDueAsync() {
        CompletableFuture.runAsync(() -> {
            try {
                Map<String, Object> res = client.post().uri("/api/retrain/maybe")
                        .retrieve()
                        .body(new ParameterizedTypeReference<Map<String, Object>>() {});
                log.info("ML retrain check: {}", res == null ? "no response" : res.getOrDefault("status", res.get("results")));
            } catch (Exception e) {
                log.warn("ML retrain check failed: {}", e.getMessage());
            }
        });
    }
}
