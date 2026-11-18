package com.screening.controller;

import com.screening.model.AssessmentReport;
import com.screening.model.AssessmentSession;
import com.screening.model.QuestionDefinition;
import com.screening.model.QuestionResponse;
import com.screening.repository.AssessmentSessionRepository;
import com.screening.model.DisorderDomain;
import com.screening.service.AssessmentScoringService;
import com.screening.service.MlServiceClient;
import com.screening.service.QuestionBankService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/assessments")
public class AssessmentController {

    private final AssessmentSessionRepository repository;
    private final QuestionBankService questionBankService;
    private final AssessmentScoringService scoringService;
    private final MlServiceClient mlClient;

    public AssessmentController(AssessmentSessionRepository repository,
                                QuestionBankService questionBankService,
                                AssessmentScoringService scoringService,
                                MlServiceClient mlClient) {
        this.repository = repository;
        this.questionBankService = questionBankService;
        this.scoringService = scoringService;
        this.mlClient = mlClient;
    }

    /**
     * Get the 15 standard screening questions with parameter metadata.
     */
    @GetMapping("/questions")
    public ResponseEntity<List<QuestionDefinition>> getQuestions() {
        return ResponseEntity.ok(questionBankService.getAllQuestions());
    }

    /**
     * Parent & child registration / session initialization.
     */
    @PostMapping("/register")
    public ResponseEntity<AssessmentSession> register(@RequestBody Map<String, Object> payload) {
        String parentName = (String) payload.getOrDefault("parentName", "Parent");
        String parentEmail = (String) payload.getOrDefault("parentEmail", "");
        String parentPhone = (String) payload.getOrDefault("parentPhone", "");
        String childName = (String) payload.getOrDefault("childName", "Student");
        int childAge = payload.get("childAge") != null ? Integer.parseInt(payload.get("childAge").toString()) : 7;
        String childGrade = (String) payload.getOrDefault("childGrade", "2nd Grade");

        AssessmentSession session = new AssessmentSession(parentName, parentEmail, parentPhone, childName, childAge, childGrade);

        // Optional parent-reported diagnoses ("yes" / "no" / "unknown") become weak training labels
        if (payload.get("knownDiagnoses") instanceof Map<?, ?> known) {
            known.forEach((disorder, answer) -> {
                String a = String.valueOf(answer);
                if (("yes".equals(a) || "no".equals(a)) && isDisorder(String.valueOf(disorder))) {
                    session.getLabels().put(String.valueOf(disorder), label("yes".equals(a) ? 1 : 0, "parent"));
                }
            });
        }
        AssessmentSession saved = repository.save(session);
        return ResponseEntity.ok(saved);
    }

    /**
     * Submit all question responses, evaluate diagnostic scores and store report in MongoDB.
     */
    @PostMapping("/{sessionId}/submit")
    public ResponseEntity<?> submitResponses(@PathVariable String sessionId,
                                             @RequestBody List<QuestionResponse> responses) {
        return repository.findById(sessionId).map(session -> {
            session.setResponses(responses);
            session.setStatus("COMPLETED");
            session.setCompletedAt(Instant.now());

            AssessmentReport report = scoringService.generateReport(responses, session.getChildAge());
            scoringService.applyModelResults(report, mlClient.score(session)); // no-op if the ML service is down
            session.setReport(report);

            AssessmentSession saved = repository.save(session);
            return ResponseEntity.ok(saved);
        }).orElseGet(() -> ResponseEntity.notFound().build());
    }

    /**
     * Fetch session details and report by ID.
     */
    @GetMapping("/{sessionId}")
    public ResponseEntity<AssessmentSession> getSession(@PathVariable String sessionId) {
        return repository.findById(sessionId)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }

    /**
     * Retrieve past assessment history.
     */
    @GetMapping("/history")
    public ResponseEntity<List<AssessmentSession>> getHistory(@RequestParam(required = false) String childName,
                                                              @RequestParam(required = false) String parentEmail) {
        if (childName != null && !childName.isBlank()) {
            return ResponseEntity.ok(repository.findByChildNameContainingIgnoreCase(childName));
        }
        if (parentEmail != null && !parentEmail.isBlank()) {
            return ResponseEntity.ok(repository.findByParentEmailIgnoreCase(parentEmail));
        }
        return ResponseEntity.ok(repository.findAllByOrderByCreatedAtDesc());
    }

    /**
     * Record a confirmed outcome for one disorder. Body: {"disorder": "DYSLEXIA", "value": 1, "source": "specialist"}.
     * A specialist label always replaces a parent-reported one, never the other way round.
     */
    @PostMapping("/{sessionId}/labels")
    public ResponseEntity<?> setLabel(@PathVariable String sessionId, @RequestBody Map<String, Object> body) {
        String disorder = String.valueOf(body.get("disorder"));
        Object v = body.get("value");
        String source = String.valueOf(body.getOrDefault("source", "specialist"));
        boolean validValue = v instanceof Number n && (n.intValue() == 0 || n.intValue() == 1);
        if (!isDisorder(disorder) || !validValue || !("specialist".equals(source) || "parent".equals(source))) {
            return ResponseEntity.badRequest().body(Map.of("error",
                    "Expected disorder DYSLEXIA, DYSGRAPHIA or DYSCALCULIA, value 0 or 1, source specialist or parent"));
        }
        int value = ((Number) v).intValue();
        return repository.findById(sessionId).<ResponseEntity<?>>map(session -> {
            Map<String, Object> existing = session.getLabels().get(disorder);
            if (existing != null && "specialist".equals(existing.get("source")) && "parent".equals(source)) {
                return ResponseEntity.status(409).body(Map.of("error", "A specialist label already exists"));
            }
            session.getLabels().put(disorder, label(value, source));
            AssessmentSession saved = repository.save(session);
            mlClient.retrainIfDueAsync();
            return ResponseEntity.ok(saved);
        }).orElseGet(() -> ResponseEntity.notFound().build());
    }

    /** Completed sessions with at least one label: the ML service's training data. */
    @GetMapping("/training-data")
    public ResponseEntity<List<AssessmentSession>> trainingData() {
        return ResponseEntity.ok(repository.findAll().stream()
                .filter(s -> "COMPLETED".equals(s.getStatus()) && s.getLabels() != null && !s.getLabels().isEmpty())
                .toList());
    }

    private static boolean isDisorder(String name) {
        try {
            DisorderDomain.valueOf(name);
            return true;
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    private static Map<String, Object> label(int value, String source) {
        Map<String, Object> l = new HashMap<>();
        l.put("value", value);
        l.put("source", source);
        l.put("at", Instant.now().toString());
        return l;
    }
}
