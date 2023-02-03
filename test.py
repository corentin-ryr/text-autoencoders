import numpy as np


SRCCMixerSmall = np.array([69.8773045164547, 78.60555812584796, 73.02980690801101, 66.74189400533453, 75.8765545495429, 67.29263237494375, 69.20692298379241, 68.21749301632202])
f1MixerSmall = np.array([0.8545, 0.8866, 0.8658, 0.8304, 0.8659, 0.8101, 0.8613, 0.8339])

print(f"SRCC of Mixer small {np.mean(SRCCMixerSmall):.2f} +- {np.std(SRCCMixerSmall):.2f}")
print(f"f1 of Mixer small {np.mean(f1MixerSmall):.4f} +- {np.std(f1MixerSmall):.4f}")

print() #========================================================================================

SRCCTransformerSmall = np.array([71.61307992963691, 65.95756660309792, 69.88110271210716, 65.69169290742448, 73.83502438633624, 66.17026555963665, 64.58261977690107, 74.78077510380312])
f1TransformerSmall = np.array([0.8482, 0.7994, 0.8561, 0.8049, 0.8656, 0.8210, 0.8152, 0.8797])

print(f"SRCC of Transformer small {np.mean(SRCCTransformerSmall):.2f} +- {np.std(SRCCTransformerSmall):.2f}")
print(f"f1 of Transformer small {np.mean(f1TransformerSmall):.4f} +- {np.std(f1TransformerSmall):.4f}")

print() #========================================================================================

SRCCTransformerMedium = np.array([67.37049538581955, 67.17488830971695, 62.60375984196028, 75.56510250603972])
f1TransformerMedium = np.array([0.8412, 0.8380, 0.8327, 0.9053])

print(f"SRCC of Transformer medium {np.mean(SRCCTransformerMedium):.2f} +- {np.std(SRCCTransformerMedium):.2f}")
print(f"f1 of Transformer medium {np.mean(f1TransformerMedium):.4f} +- {np.std(f1TransformerMedium):.4f}")