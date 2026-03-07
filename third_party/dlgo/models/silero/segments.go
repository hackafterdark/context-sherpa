package silero

import "math"

// VADParams controls speech segment detection from probabilities.
type VADParams struct {
	Threshold            float32 // probability ≥ this = speech (default 0.5)
	MinSpeechDurationMs  int     // discard segments shorter than this (default 250)
	MinSilenceDurationMs int     // need this much silence to split (default 100)
	MaxSpeechDurationS   float32 // auto-split segments longer than this (0 = unlimited)
	SpeechPadMs          int     // extend segments by this on each side (default 30)
}

// DefaultVADParams returns the same defaults as whisper-cpp.
func DefaultVADParams() VADParams {
	return VADParams{
		Threshold:            0.5,
		MinSpeechDurationMs:  250,
		MinSilenceDurationMs: 100,
		MaxSpeechDurationS:   0,
		SpeechPadMs:          30,
	}
}

// SpeechSegment represents a detected speech region.
type SpeechSegment struct {
	StartSample int     // exact start sample (16kHz)
	EndSample   int     // exact end sample (16kHz)
	StartCs     int64   // start time in centiseconds (matches whisper-cpp)
	EndCs       int64   // end time in centiseconds
	StartS      float32 // start time in seconds (convenience)
	EndS        float32 // end time in seconds (convenience)
}

// DetectSegments converts per-chunk probabilities into speech segments.
// Matches whisper-cpp's whisper_vad_segments_from_probs algorithm:
//   - Dual threshold (neg_threshold = threshold - 0.15) for hysteresis
//   - Sample-based counting
//   - 200ms post-processing merge
//   - Complex padding with gap splitting
func DetectSegments(probs []float32, nWindow int, params VADParams) []SpeechSegment {
	const sampleRate = 16000

	nProbs := len(probs)
	minSilenceSamples := sampleRate * params.MinSilenceDurationMs / 1000
	minSpeechSamples := sampleRate * params.MinSpeechDurationMs / 1000
	speechPadSamples := sampleRate * params.SpeechPadMs / 1000
	audioLengthSamples := nProbs * nWindow

	// Max speech duration
	maxSpeechSamples := math.MaxInt32 / 2
	if params.MaxSpeechDurationS > 0 && params.MaxSpeechDurationS < 100000.0 {
		temp := int(float64(sampleRate)*float64(params.MaxSpeechDurationS)) - nWindow - 2*speechPadSamples
		if temp > 0 {
			maxSpeechSamples = temp
		}
	}

	// From silero-vad python: 98ms silence threshold for max_speech splitting
	minSilenceSamplesAtMaxSpeech := sampleRate * 98 / 1000

	// Negative threshold for end-of-speech detection (hysteresis)
	negThreshold := params.Threshold - 0.15
	if negThreshold < 0.01 {
		negThreshold = 0.01
	}

	type speechSeg struct {
		start, end int
	}

	var speeches []speechSeg
	isSpeechSegment := false
	tempEnd := 0
	prevEnd := 0
	nextStart := 0
	currSpeechStart := 0
	hasCurrSpeech := false

	for i := 0; i < nProbs; i++ {
		currProb := probs[i]
		currSample := nWindow * i

		// Reset temp_end when we get back to speech
		if currProb >= params.Threshold && tempEnd != 0 {
			tempEnd = 0
			if nextStart < prevEnd {
				nextStart = currSample
			}
		}

		// Start new speech segment
		if currProb >= params.Threshold && !isSpeechSegment {
			isSpeechSegment = true
			currSpeechStart = currSample
			hasCurrSpeech = true
			continue
		}

		// Handle maximum speech duration
		if isSpeechSegment && (currSample-currSpeechStart) > maxSpeechSamples {
			if prevEnd != 0 {
				speeches = append(speeches, speechSeg{currSpeechStart, prevEnd})
				hasCurrSpeech = true

				if nextStart < prevEnd {
					isSpeechSegment = false
					hasCurrSpeech = false
				} else {
					currSpeechStart = nextStart
				}
				prevEnd = 0
				nextStart = 0
				tempEnd = 0
			} else {
				speeches = append(speeches, speechSeg{currSpeechStart, currSample})
				prevEnd = 0
				nextStart = 0
				tempEnd = 0
				isSpeechSegment = false
				hasCurrSpeech = false
				continue
			}
		}

		// Handle silence after speech
		if currProb < negThreshold && isSpeechSegment {
			if tempEnd == 0 {
				tempEnd = currSample
			}

			// Track potential segment ends for max_speech handling
			if (currSample - tempEnd) > minSilenceSamplesAtMaxSpeech {
				prevEnd = tempEnd
			}

			// Check if silence is long enough to end the segment
			if (currSample - tempEnd) < minSilenceSamples {
				continue
			}

			// End the segment if it's long enough
			if (tempEnd - currSpeechStart) > minSpeechSamples {
				speeches = append(speeches, speechSeg{currSpeechStart, tempEnd})
			}

			prevEnd = 0
			nextStart = 0
			tempEnd = 0
			isSpeechSegment = false
			hasCurrSpeech = false
			continue
		}
	}

	// Handle speech at end of audio
	if hasCurrSpeech && (audioLengthSamples-currSpeechStart) > minSpeechSamples {
		speeches = append(speeches, speechSeg{currSpeechStart, audioLengthSamples})
	}

	// Post-processing: merge adjacent segments with small gaps (200ms)
	if len(speeches) > 1 {
		maxMergeGapSamples := sampleRate * 200 / 1000
		for i := 0; i < len(speeches)-1; i++ {
			if speeches[i+1].start-speeches[i].end < maxMergeGapSamples {
				speeches[i].end = speeches[i+1].end
				speeches = append(speeches[:i+1], speeches[i+2:]...)
				i--
			}
		}
	}

	// Remove segments shorter than min_speech_duration
	for i := 0; i < len(speeches); i++ {
		if speeches[i].end-speeches[i].start < minSpeechSamples {
			speeches = append(speeches[:i], speeches[i+1:]...)
			i--
		}
	}

	// Apply padding and convert to centiseconds
	segments := make([]SpeechSegment, len(speeches))
	for i := 0; i < len(speeches); i++ {
		if i == 0 {
			if speeches[i].start > speechPadSamples {
				speeches[i].start -= speechPadSamples
			} else {
				speeches[i].start = 0
			}
		}

		if i < len(speeches)-1 {
			silenceDuration := speeches[i+1].start - speeches[i].end
			if silenceDuration < 2*speechPadSamples {
				speeches[i].end += silenceDuration / 2
				if speeches[i+1].start > silenceDuration/2 {
					speeches[i+1].start -= silenceDuration / 2
				} else {
					speeches[i+1].start = 0
				}
			} else {
				if speeches[i].end+speechPadSamples < audioLengthSamples {
					speeches[i].end += speechPadSamples
				} else {
					speeches[i].end = audioLengthSamples
				}
				if speeches[i+1].start > speechPadSamples {
					speeches[i+1].start -= speechPadSamples
				} else {
					speeches[i+1].start = 0
				}
			}
		} else {
			if speeches[i].end+speechPadSamples < audioLengthSamples {
				speeches[i].end += speechPadSamples
			} else {
				speeches[i].end = audioLengthSamples
			}
		}

		startCs := samplesToCs(speeches[i].start)
		endCs := samplesToCs(speeches[i].end)
		segments[i] = SpeechSegment{
			StartSample: speeches[i].start,
			EndSample:   speeches[i].end,
			StartCs:     startCs,
			EndCs:       endCs,
			StartS:      float32(startCs) / 100.0,
			EndS:        float32(endCs) / 100.0,
		}
	}

	return segments
}

func samplesToCs(samples int) int64 {
	return int64(float64(samples)/float64(16000)*100.0 + 0.5)
}
