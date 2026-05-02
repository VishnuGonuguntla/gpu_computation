# Compiler & flags
CC         = g++
NVCC       = nvcc
C_FLAGS    = -O3 -march=native -std=c++17
CUDA_FLAGS = -O3 -std=c++17

SRC_DIR   = src
BUILD_DIR = build

.DEFAULT_GOAL := all
# ── Auto-discover exercises (subdirs of src/) ────────────────────────────────
EXERCISES := $(shell find $(SRC_DIR) -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)

# ── Per-file binary macro ────────────────────────────────────────────────────
# $(1) = source file, $(2) = output binary
define FILE_RULES
$(2): $(1)
	@mkdir -p $$(@D)
	$(if $(filter %.cu,$(1)),$(NVCC) $(CUDA_FLAGS),$(CC) $(C_FLAGS)) $(1) -o $(2)
endef

# ── Per-exercise macro ───────────────────────────────────────────────────────
define EXERCISE_RULES

$(1)_SRCS      := $$(shell find $(SRC_DIR)/$(1) -type f \( -name '*.cpp' -o -name '*.cu' \))
$(1)_CPP_SRCS  := $$(filter %.cpp,$$($(1)_SRCS))
$(1)_CU_SRCS   := $$(filter %.cu,$$($(1)_SRCS))
$(1)_CPP_STEMS := $$(patsubst $(SRC_DIR)/$(1)/%.cpp,%,$$($(1)_CPP_SRCS))
$(1)_CU_STEMS  := $$(patsubst $(SRC_DIR)/$(1)/%.cu,%,$$($(1)_CU_SRCS))
$(1)_SHARED    := $$(filter $$($(1)_CU_STEMS),$$($(1)_CPP_STEMS))

$(1)_CPP_BINS  := $$(foreach stem,$$($(1)_CPP_STEMS),$(BUILD_DIR)/$(1)/$$(if $$(filter $$(stem),$$($(1)_SHARED)),$$(stem)_cpp,$$(stem)))
$(1)_CU_BINS   := $$(foreach stem,$$($(1)_CU_STEMS),$(BUILD_DIR)/$(1)/$$(if $$(filter $$(stem),$$($(1)_SHARED)),$$(stem)_cu,$$(stem)))
$(1)_BINS      := $$($(1)_CPP_BINS) $$($(1)_CU_BINS)

.PHONY: $(1)
$(1): $$($(1)_BINS)

$$(foreach stem,$$($(1)_CPP_STEMS),$$(eval $$(call FILE_RULES,$(SRC_DIR)/$(1)/$$(stem).cpp,$(BUILD_DIR)/$(1)/$$(if $$(filter $$(stem),$$($(1)_SHARED)),$$(stem)_cpp,$$(stem)))))
$$(foreach stem,$$($(1)_CU_STEMS),$$(eval $$(call FILE_RULES,$(SRC_DIR)/$(1)/$$(stem).cu,$(BUILD_DIR)/$(1)/$$(if $$(filter $$(stem),$$($(1)_SHARED)),$$(stem)_cu,$$(stem)))))

endef

# ── Instantiate rules for every exercise ────────────────────────────────────
$(foreach ex, $(EXERCISES), $(eval $(call EXERCISE_RULES,$(ex))))

# ── Convenience targets ──────────────────────────────────────────────────────
.PHONY: all clean

all: $(EXERCISES)

clean:
	rm -rf $(BUILD_DIR)