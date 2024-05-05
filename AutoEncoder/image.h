#pragma once
#include <vector>

using namespace std;
vector<vector<vector<vector<long double>>>> image = {
    {{{-0.606833, -0.963339, -0.923727, -1.002951, -0.745474, -0.527609, -0.448386, -0.448386, -0.349356, -0.468192, -0.567221, -0.547415, -0.567221, -0.626639, -0.646445, -0.527609, -0.567221, -0.547415, -0.666251, -0.587027, -0.408774, -0.349356, -0.408774, -0.388968, -0.369162, -0.408774, -0.527609, -0.388968, 0.185403, 0.720163, 0.997445, 1.136086},
      {-0.626639, -0.963339, -0.923727, -0.943533, -0.686057, -0.428580, -0.270133, -0.349356, -0.567221, -0.587027, -0.507803, -0.388968, -0.349356, -0.428580, -0.408774, -0.408774, -0.507803, -0.587027, -0.765280, -0.765280, -0.487998, -0.369162, -0.408774, -0.408774, -0.369162, -0.408774, -0.547415, -0.428580, 0.125985, 0.660745, 0.977639, 1.136086},
      {-0.646445, -0.943533, -0.884116, -0.844504, -0.587027, -0.270133, -0.012656, -0.171103, -0.547415, -0.428580, -0.111685, 0.125985, 0.145791, 0.145791, 0.145791, 0.086374, -0.171103, -0.369162, -0.646445, -0.626639, -0.329550, -0.111685, -0.190909, -0.190909, -0.309744, -0.369162, -0.547415, -0.487998, 0.046762, 0.561715, 0.918222, 1.096475},
      {-0.626639, -0.884116, -0.804892, -0.705862, -0.428580, -0.072074, 0.304239, 0.066568, -0.250327, -0.032462, 0.462686, 0.779580, 0.759774, 0.739968, 0.700357, 0.522104, 0.165597, -0.111685, -0.487998, -0.468192, -0.032462, 0.205209, 0.106180, 0.026956, -0.210715, -0.250327, -0.468192, -0.487998, -0.052268, 0.502298, 0.878610, 1.076669},
      {-0.587027, -0.785086, -0.686057, -0.547415, -0.270133, 0.165597, 0.660745, 0.284433, -0.111685, 0.185403, 0.759774, 1.076669, 1.037057, 1.037057, 1.037057, 0.799386, 0.264627, -0.091880, -0.468192, -0.408774, 0.086374, 0.324045, 0.165597, 0.007150, -0.012656, -0.091880, -0.388968, -0.448386, -0.072074, 0.403268, 0.799386, 1.076669},
      {-0.547415, -0.705862, -0.606833, -0.428580, -0.131491, 0.363656, 0.898416, 0.502298, -0.111685, 0.145791, 0.660745, 0.918222, 0.799386, 0.918222, 1.017251, 0.799386, 0.244821, -0.151297, -0.587027, -0.448386, 0.106180, 0.343850, 0.125985, -0.131491, 0.145791, 0.066568, -0.270133, -0.428580, -0.131491, 0.343850, 0.759774, 1.056863},
      {-0.507803, -0.646445, -0.547415, -0.349356, -0.052268, 0.462686, 1.076669, 0.601327, -0.012656, 0.145791, 0.541909, 0.680551, 0.442880, 0.561715, 0.680551, 0.482492, 0.185403, -0.210715, -0.646445, -0.527609, 0.086374, 0.343850, 0.066568, -0.230521, 0.225015, 0.125985, -0.210715, -0.408774, -0.151297, 0.343850, 0.739968, 1.037057},
      {-0.587027, -0.626639, -0.567221, -0.428580, -0.012656, 0.522104, 1.037057, 0.739968, -0.309744, -0.408774, -0.131491, -0.012656, -0.190909, -0.052268, 0.145791, -0.032462, -0.289939, -0.547415, -0.884116, -0.606833, 0.125985, 0.225015, -0.131491, -0.289939, 0.363656, 0.423074, 0.007150, -0.270133, -0.091880, 0.324045, 0.779580, 0.918222},
      {-0.487998, -0.606833, -0.567221, -0.428580, -0.032462, 0.462686, 0.997445, 0.700357, -0.190909, -0.428580, -0.408774, -0.369162, -0.626639, -0.507803, -0.329550, -0.507803, -0.686057, -0.844504, -1.062369, -0.666251, 0.086374, 0.185403, -0.190909, -0.289939, 0.324045, 0.363656, -0.052268, -0.329550, -0.131491, 0.264627, 0.720163, 0.898416},
      {-0.388968, -0.567221, -0.587027, -0.448386, -0.091880, 0.383462, 0.918222, 0.660745, -0.151297, -0.547415, -0.666251, -0.745474, -1.042563, -0.864310, -0.587027, -0.686057, -0.765280, -0.923727, -1.042563, -0.606833, 0.106180, 0.225015, -0.072074, -0.210715, 0.225015, 0.244821, -0.171103, -0.388968, -0.210715, 0.205209, 0.680551, 0.898416},
      {-0.309744, -0.527609, -0.587027, -0.507803, -0.151297, 0.304239, 0.838998, 0.621133, -0.131491, -0.666251, -0.923727, -1.002951, -1.240622, -1.022757, -0.725668, -0.785086, -0.844504, -1.042563, -1.161398, -0.804892, -0.190909, -0.012656, -0.210715, -0.270133, 0.145791, 0.125985, -0.309744, -0.527609, -0.309744, 0.086374, 0.601327, 0.878610},
      {-0.230521, -0.527609, -0.646445, -0.547415, -0.230521, 0.205209, 0.799386, 0.621133, -0.111685, -0.765280, -1.062369, -1.121786, -1.280234, -1.121786, -0.785086, -0.864310, -0.864310, -1.121786, -1.339651, -1.082175, -0.547415, -0.270133, -0.250327, -0.230521, 0.086374, 0.026956, -0.428580, -0.587027, -0.388968, -0.012656, 0.541909, 0.878610},
      {-0.171103, -0.567221, -0.725668, -0.606833, -0.270133, 0.145791, 0.759774, 0.660745, -0.072074, -0.785086, -1.101981, -1.062369, -1.082175, -0.824698, -0.487998, -0.507803, -0.785086, -1.161398, -1.399069, -1.201010, -0.765280, -0.408774, -0.190909, -0.072074, 0.046762, -0.091880, -0.547415, -0.686057, -0.468192, -0.091880, 0.482492, 0.878610},
      {-0.190909, -0.626639, -0.785086, -0.666251, -0.309744, 0.106180, 0.759774, 0.720163, 0.086374, -0.745474, -1.082175, -0.943533, -0.884116, -0.527609, -0.111685, -0.052268, -0.606833, -1.002951, -1.280234, -1.181204, -0.844504, -0.487998, -0.230521, -0.032462, -0.032462, -0.131491, -0.626639, -0.785086, -0.527609, -0.131491, 0.442880, 0.878610},
      {-0.190909, -0.626639, -0.824698, -0.666251, -0.329550, 0.086374, 0.759774, 0.759774, 0.225015, -0.646445, -1.042563, -0.943533, -0.864310, -0.507803, -0.091880, 0.007150, -0.171103, -0.567221, -0.903921, -0.864310, -0.646445, -0.329550, -0.072074, 0.046762, -0.091880, -0.250327, -0.725668, -0.785086, -0.547415, -0.171103, 0.423074, 0.878610},
      {-0.032462, -0.606833, -0.785086, -0.804892, -0.785086, -0.111685, 0.680551, 0.838998, 0.324045, -0.547415, -0.824698, -0.903921, -1.240622, -1.300040, -0.844504, -0.270133, 0.145791, 0.363656, -0.111685, -0.428580, -0.171103, 0.086374, -0.012656, -0.487998, -0.487998, -0.270133, -0.131491, -0.230521, -0.428580, -0.289939, 0.225015, 0.759774},
      {-0.032462, -0.587027, -0.785086, -0.824698, -0.785086, -0.151297, 0.621133, 0.799386, 0.165597, -0.587027, -0.824698, -0.864310, -1.201010, -1.280234, -0.983145, -0.448386, 0.125985, 0.423074, 0.125985, -0.072074, 0.145791, 0.343850, 0.086374, -0.487998, -0.527609, -0.250327, -0.052268, -0.131491, -0.349356, -0.250327, 0.225015, 0.739968},
      {0.007150, -0.547415, -0.785086, -0.824698, -0.804892, -0.210715, 0.522104, 0.779580, 0.205209, -0.468192, -0.527609, -0.507803, -0.824698, -1.022757, -0.884116, -0.468192, 0.046762, 0.383462, 0.244821, 0.125985, 0.304239, 0.423074, 0.106180, -0.567221, -0.587027, -0.270133, 0.026956, 0.007150, -0.230521, -0.190909, 0.264627, 0.739968},
      {0.007150, -0.468192, -0.765280, -0.864310, -0.844504, -0.309744, 0.442880, 0.700357, 0.343850, -0.151297, -0.131491, -0.091880, -0.388968, -0.705862, -0.725668, -0.428580, 0.145791, 0.502298, 0.442880, 0.324045, 0.442880, 0.601327, 0.264627, -0.428580, -0.666251, -0.270133, 0.106180, 0.145791, -0.111685, -0.131491, 0.244821, 0.680551},
      {0.046762, -0.428580, -0.745474, -0.923727, -0.864310, -0.349356, 0.324045, 0.581521, 0.304239, -0.052268, 0.007150, 0.106180, -0.230521, -0.567221, -0.725668, -0.487998, 0.125985, 0.502298, 0.522104, 0.442880, 0.502298, 0.621133, 0.304239, -0.448386, -0.765280, -0.289939, 0.165597, 0.205209, -0.052268, -0.131491, 0.225015, 0.621133},
      {0.066568, -0.349356, -0.785086, -0.943533, -0.864310, -0.388968, 0.244821, 0.502298, 0.304239, 0.026956, 0.106180, 0.125985, -0.151297, -0.547415, -0.785086, -0.587027, -0.052268, 0.343850, 0.502298, 0.462686, 0.442880, 0.502298, 0.205209, -0.587027, -0.844504, -0.329550, 0.165597, 0.205209, -0.072074, -0.171103, 0.165597, 0.581521},
      {0.086374, -0.369162, -0.765280, -1.002951, -0.903921, -0.428580, 0.145791, 0.442880, 0.343850, 0.165597, 0.165597, 0.125985, -0.171103, -0.547415, -0.804892, -0.606833, 0.046762, 0.383462, 0.482492, 0.423074, 0.383462, 0.482492, 0.264627, -0.487998, -0.903921, -0.369162, 0.145791, 0.225015, -0.091880, -0.190909, 0.125985, 0.561715},
      {0.106180, -0.329550, -0.765280, -1.002951, -0.903921, -0.448386, 0.106180, 0.423074, 0.284433, 0.086374, 0.066568, -0.012656, -0.250327, -0.626639, -0.884116, -0.686057, 0.026956, 0.304239, 0.324045, 0.205209, 0.106180, 0.304239, 0.225015, -0.468192, -0.903921, -0.388968, 0.145791, 0.205209, -0.131491, -0.230521, 0.086374, 0.522104},
      {0.225015, -0.131491, -0.507803, -0.725668, -0.785086, -0.428580, 0.066568, 0.363656, 0.304239, 0.086374, -0.131491, -0.131491, -0.270133, -0.705862, -0.824698, -0.567221, -0.032462, 0.145791, -0.032462, -0.230521, -0.309744, 0.007150, 0.225015, -0.270133, -0.765280, -0.487998, 0.026956, 0.007150, -0.309744, -0.369162, -0.151297, 0.442880},
      {0.284433, -0.072074, -0.448386, -0.666251, -0.765280, -0.428580, 0.066568, 0.324045, 0.244821, 0.007150, -0.230521, -0.230521, -0.369162, -0.745474, -0.824698, -0.587027, -0.111685, 0.007150, -0.210715, -0.428580, -0.487998, -0.151297, 0.066568, -0.329550, -0.745474, -0.527609, -0.091880, -0.111685, -0.448386, -0.468192, -0.190909, 0.423074},
      {0.383462, 0.026956, -0.349356, -0.587027, -0.705862, -0.428580, 0.007150, 0.225015, 0.086374, -0.111685, -0.369162, -0.408774, -0.507803, -0.785086, -0.903921, -0.646445, -0.289939, -0.250327, -0.487998, -0.785086, -0.765280, -0.408774, -0.171103, -0.428580, -0.745474, -0.567221, -0.270133, -0.329550, -0.626639, -0.606833, -0.210715, 0.383462},
      {0.482492, 0.125985, -0.210715, -0.507803, -0.666251, -0.487998, -0.151297, 0.046762, -0.111685, -0.309744, -0.567221, -0.606833, -0.626639, -0.844504, -0.943533, -0.765280, -0.487998, -0.527609, -0.804892, -1.101981, -1.062369, -0.705862, -0.487998, -0.587027, -0.785086, -0.646445, -0.468192, -0.567221, -0.824698, -0.745474, -0.250327, 0.343850},
      {0.502298, 0.205209, -0.190909, -0.507803, -0.725668, -0.646445, -0.369162, -0.210715, -0.388968, -0.527609, -0.765280, -0.785086, -0.765280, -0.903921, -1.002951, -0.884116, -0.725668, -0.804892, -1.042563, -1.280234, -1.201010, -0.903921, -0.765280, -0.725668, -0.844504, -0.745474, -0.666251, -0.725668, -0.903921, -0.824698, -0.230521, 0.403268},
      {0.502298, 0.125985, -0.210715, -0.547415, -0.804892, -0.785086, -0.587027, -0.487998, -0.646445, -0.765280, -0.963339, -0.943533, -0.864310, -0.943533, -1.101981, -1.062369, -0.943533, -1.042563, -1.181204, -1.339651, -1.240622, -1.042563, -1.042563, -0.923727, -1.002951, -0.844504, -0.804892, -0.844504, -0.963339, -0.844504, -0.171103, 0.442880},
      {0.442880, 0.106180, -0.270133, -0.606833, -0.923727, -0.943533, -0.804892, -0.765280, -0.884116, -0.963339, -1.082175, -1.022757, -0.884116, -0.983145, -1.181204, -1.181204, -1.082175, -1.161398, -1.220816, -1.260428, -1.141592, -1.022757, -1.201010, -1.022757, -1.101981, -0.943533, -0.923727, -0.903921, -0.943533, -0.804892, -0.091880, 0.442880},
      {0.383462, 0.046762, -0.309744, -0.626639, -0.983145, -1.022757, -0.903921, -0.903921, -1.002951, -1.042563, -1.161398, -1.062369, -0.903921, -0.963339, -1.201010, -1.220816, -1.161398, -1.220816, -1.201010, -1.201010, -1.022757, -0.983145, -1.240622, -1.042563, -1.121786, -0.943533, -0.963339, -0.903921, -0.923727, -0.804892, -0.052268, 0.502298},
      {0.343850, 0.106180, -0.250327, -0.606833, -0.824698, -0.864310, -0.804892, -0.725668, -0.804892, -0.844504, -0.844504, -0.884116, -0.884116, -0.884116, -0.923727, -0.903921, -0.844504, -0.844504, -0.884116, -0.824698, -0.884116, -0.844504, -0.903921, -0.903921, -0.646445, -0.725668, -0.765280, -0.686057, -0.507803, -0.230521, 0.086374, 0.284433}},
     {{-0.627627, -0.978812, -0.959301, -1.037343, -0.764199, -0.549585, -0.471544, -0.491054, -0.393503, -0.510565, -0.608116, -0.588606, -0.588606, -0.647137, -0.647137, -0.549585, -0.588606, -0.608116, -0.764199, -0.744688, -0.569096, -0.549585, -0.608116, -0.588606, -0.549585, -0.588606, -0.686157, -0.549585, 0.035723, 0.582012, 0.874666, 1.011238},
      {-0.647137, -0.998322, -0.959301, -0.978812, -0.725178, -0.452034, -0.276441, -0.393503, -0.608116, -0.627627, -0.549585, -0.413013, -0.373993, -0.413013, -0.413013, -0.373993, -0.530075, -0.627627, -0.822729, -0.861750, -0.647137, -0.569096, -0.588606, -0.588606, -0.549585, -0.588606, -0.725178, -0.588606, -0.042318, 0.542991, 0.855156, 1.011238},
      {-0.686157, -0.959301, -0.920281, -0.900771, -0.647137, -0.295952, -0.022808, -0.217910, -0.569096, -0.452034, -0.139869, 0.133275, 0.152785, 0.152785, 0.152785, 0.074744, -0.178890, -0.393503, -0.705668, -0.725178, -0.432524, -0.256931, -0.334972, -0.373993, -0.491054, -0.510565, -0.686157, -0.608116, -0.081338, 0.445440, 0.796625, 1.030748},
      {-0.666647, -0.900771, -0.842240, -0.764199, -0.491054, -0.100849, 0.289357, 0.035723, -0.276441, -0.061828, 0.445440, 0.777115, 0.757604, 0.738094, 0.699073, 0.503970, 0.152785, -0.139869, -0.510565, -0.530075, -0.120359, 0.055234, -0.022808, -0.159380, -0.354482, -0.393503, -0.608116, -0.608116, -0.178890, 0.386909, 0.757604, 1.011238},
      {-0.627627, -0.842240, -0.744688, -0.608116, -0.334972, 0.113765, 0.601522, 0.250337, -0.139869, 0.172295, 0.738094, 1.089279, 1.050259, 1.089279, 1.089279, 0.835645, 0.289357, -0.061828, -0.491054, -0.471544, 0.016213, 0.230826, 0.074744, -0.139869, -0.159380, -0.237421, -0.471544, -0.569096, -0.198400, 0.328378, 0.738094, 1.011238},
      {-0.588606, -0.764199, -0.666647, -0.491054, -0.178890, 0.308868, 0.855156, 0.464950, -0.139869, 0.133275, 0.660053, 0.913687, 0.855156, 0.972217, 1.069769, 0.835645, 0.269847, -0.120359, -0.569096, -0.471544, 0.035723, 0.289357, 0.035723, -0.217910, 0.055234, -0.022808, -0.354482, -0.510565, -0.198400, 0.269847, 0.699073, 0.991728},
      {-0.549585, -0.705668, -0.608116, -0.413013, -0.100849, 0.406419, 1.030748, 0.562501, -0.042318, 0.152785, 0.542991, 0.679563, 0.503970, 0.601522, 0.757604, 0.542991, 0.230826, -0.159380, -0.627627, -0.510565, 0.035723, 0.308868, -0.003297, -0.295952, 0.152785, 0.074744, -0.256931, -0.452034, -0.178890, 0.308868, 0.718584, 1.011238},
      {-0.627627, -0.666647, -0.627627, -0.491054, -0.061828, 0.464950, 0.991728, 0.699073, -0.334972, -0.413013, -0.139869, 0.035723, -0.139869, 0.016213, 0.211316, 0.016213, -0.237421, -0.510565, -0.842240, -0.608116, 0.133275, 0.230826, -0.159380, -0.315462, 0.289357, 0.367398, -0.042318, -0.315462, -0.120359, 0.289357, 0.757604, 0.894176},
      {-0.549585, -0.647137, -0.608116, -0.510565, -0.120359, 0.406419, 0.952707, 0.640543, -0.237421, -0.432524, -0.373993, -0.334972, -0.588606, -0.452034, -0.217910, -0.413013, -0.588606, -0.764199, -1.037343, -0.666647, 0.074744, 0.230826, -0.139869, -0.315462, 0.289357, 0.328378, -0.081338, -0.334972, -0.139869, 0.250337, 0.718584, 0.894176},
      {-0.452034, -0.608116, -0.647137, -0.530075, -0.178890, 0.328378, 0.855156, 0.601522, -0.178890, -0.569096, -0.647137, -0.725178, -0.978812, -0.803219, -0.491054, -0.608116, -0.686157, -0.842240, -0.959301, -0.588606, 0.133275, 0.308868, -0.022808, -0.198400, 0.230826, 0.269847, -0.139869, -0.413013, -0.217910, 0.191806, 0.679563, 0.894176},
      {-0.393503, -0.608116, -0.666647, -0.588606, -0.237421, 0.211316, 0.738094, 0.562501, -0.159380, -0.705668, -0.920281, -0.998322, -1.193425, -0.978812, -0.647137, -0.686157, -0.744688, -0.939791, -1.095874, -0.725178, -0.120359, 0.074744, -0.120359, -0.217910, 0.191806, 0.172295, -0.256931, -0.491054, -0.276441, 0.133275, 0.640543, 0.874666},
      {-0.315462, -0.608116, -0.744688, -0.627627, -0.315462, 0.113765, 0.699073, 0.542991, -0.159380, -0.803219, -1.056853, -1.115384, -1.251956, -1.037343, -0.705668, -0.764199, -0.764199, -1.037343, -1.232446, -0.978812, -0.471544, -0.178890, -0.139869, -0.178890, 0.133275, 0.074744, -0.373993, -0.549585, -0.354482, 0.035723, 0.582012, 0.874666},
      {-0.315462, -0.647137, -0.822729, -0.705668, -0.354482, 0.055234, 0.660053, 0.582012, -0.120359, -0.842240, -1.095874, -1.037343, -1.056853, -0.744688, -0.393503, -0.413013, -0.705668, -1.037343, -1.251956, -1.095874, -0.647137, -0.276441, -0.081338, -0.022808, 0.055234, -0.042318, -0.491054, -0.627627, -0.393503, -0.061828, 0.523481, 0.874666},
      {-0.334972, -0.725178, -0.881260, -0.764199, -0.393503, 0.016213, 0.660053, 0.640543, -0.003297, -0.803219, -1.095874, -0.959301, -0.861750, -0.471544, -0.022808, -0.003297, -0.510565, -0.861750, -1.134894, -1.056853, -0.725178, -0.334972, -0.120359, 0.016213, 0.016213, -0.081338, -0.569096, -0.725178, -0.452034, -0.100849, 0.484460, 0.855156},
      {-0.334972, -0.725178, -0.920281, -0.764199, -0.432524, 0.035723, 0.699073, 0.640543, 0.113765, -0.744688, -1.134894, -1.037343, -0.920281, -0.491054, -0.042318, 0.016213, -0.100849, -0.491054, -0.783709, -0.744688, -0.510565, -0.217910, 0.035723, 0.152785, 0.016213, -0.159380, -0.627627, -0.725178, -0.491054, -0.139869, 0.464950, 0.855156},
      {-0.178890, -0.705668, -0.881260, -0.900771, -0.842240, -0.159380, 0.621032, 0.718584, 0.211316, -0.666647, -0.959301, -1.037343, -1.329997, -1.310487, -0.822729, -0.256931, 0.172295, 0.386909, -0.022808, -0.334972, -0.061828, 0.191806, 0.094254, -0.373993, -0.393503, -0.178890, -0.042318, -0.178890, -0.373993, -0.256931, 0.269847, 0.738094},
      {-0.178890, -0.686157, -0.881260, -0.881260, -0.842240, -0.178890, 0.582012, 0.718584, 0.055234, -0.744688, -0.939791, -0.978812, -1.290976, -1.329997, -0.978812, -0.452034, 0.152785, 0.464950, 0.172295, -0.022808, 0.211316, 0.406419, 0.152785, -0.432524, -0.432524, -0.159380, 0.035723, -0.081338, -0.295952, -0.217910, 0.269847, 0.718584},
      {-0.139869, -0.647137, -0.842240, -0.881260, -0.822729, -0.178890, 0.542991, 0.718584, 0.113765, -0.569096, -0.627627, -0.627627, -0.900771, -1.076363, -0.881260, -0.452034, 0.055234, 0.445440, 0.308868, 0.172295, 0.347888, 0.464950, 0.133275, -0.510565, -0.491054, -0.139869, 0.152785, 0.035723, -0.178890, -0.159380, 0.250337, 0.699073},
      {-0.081338, -0.569096, -0.822729, -0.900771, -0.822729, -0.198400, 0.503970, 0.660053, 0.289357, -0.217910, -0.198400, -0.178890, -0.432524, -0.686157, -0.666647, -0.373993, 0.211316, 0.562501, 0.523481, 0.367398, 0.484460, 0.601522, 0.308868, -0.373993, -0.569096, -0.139869, 0.230826, 0.172295, -0.061828, -0.100849, 0.230826, 0.640543},
      {-0.042318, -0.491054, -0.764199, -0.900771, -0.803219, -0.198400, 0.445440, 0.640543, 0.328378, -0.022808, 0.035723, 0.074744, -0.217910, -0.510565, -0.666647, -0.413013, 0.191806, 0.582012, 0.601522, 0.503970, 0.562501, 0.621032, 0.347888, -0.373993, -0.647137, -0.159380, 0.289357, 0.289357, -0.003297, -0.100849, 0.211316, 0.582012},
      {0.016213, -0.373993, -0.725178, -0.881260, -0.744688, -0.237421, 0.406419, 0.621032, 0.367398, 0.094254, 0.172295, 0.152785, -0.100849, -0.432524, -0.666647, -0.452034, 0.094254, 0.484460, 0.601522, 0.523481, 0.503970, 0.562501, 0.250337, -0.510565, -0.725178, -0.198400, 0.289357, 0.289357, -0.022808, -0.139869, 0.152785, 0.542991},
      {0.035723, -0.334972, -0.705668, -0.881260, -0.744688, -0.237421, 0.347888, 0.601522, 0.484460, 0.250337, 0.230826, 0.191806, -0.061828, -0.393503, -0.647137, -0.471544, 0.191806, 0.523481, 0.640543, 0.523481, 0.445440, 0.542991, 0.308868, -0.413013, -0.783709, -0.237421, 0.269847, 0.269847, -0.061828, -0.217910, 0.113765, 0.484460},
      {0.074744, -0.276441, -0.686157, -0.881260, -0.744688, -0.256931, 0.308868, 0.582012, 0.445440, 0.211316, 0.191806, 0.094254, -0.139869, -0.471544, -0.725178, -0.530075, 0.191806, 0.484460, 0.464950, 0.289357, 0.191806, 0.347888, 0.269847, -0.393503, -0.842240, -0.295952, 0.230826, 0.250337, -0.100849, -0.237421, 0.074744, 0.445440},
      {0.191806, -0.139869, -0.432524, -0.647137, -0.627627, -0.237421, 0.250337, 0.523481, 0.464950, 0.211316, -0.003297, -0.022808, -0.159380, -0.569096, -0.666647, -0.413013, 0.133275, 0.308868, 0.152785, -0.139869, -0.217910, 0.055234, 0.250337, -0.217910, -0.705668, -0.393503, 0.113765, 0.016213, -0.276441, -0.373993, -0.159380, 0.367398},
      {0.250337, -0.081338, -0.373993, -0.588606, -0.608116, -0.237421, 0.250337, 0.484460, 0.367398, 0.133275, -0.120359, -0.120359, -0.276441, -0.608116, -0.686157, -0.432524, 0.035723, 0.172295, -0.042318, -0.354482, -0.413013, -0.120359, 0.094254, -0.315462, -0.686157, -0.432524, -0.003297, -0.100849, -0.413013, -0.491054, -0.198400, 0.347888},
      {0.367398, 0.035723, -0.295952, -0.510565, -0.569096, -0.256931, 0.172295, 0.347888, 0.211316, -0.042318, -0.315462, -0.373993, -0.413013, -0.686157, -0.783709, -0.510565, -0.139869, -0.100849, -0.373993, -0.666647, -0.686157, -0.373993, -0.139869, -0.413013, -0.686157, -0.471544, -0.159380, -0.315462, -0.588606, -0.627627, -0.256931, 0.308868},
      {0.425929, 0.094254, -0.217910, -0.471544, -0.588606, -0.315462, 0.016213, 0.113765, -0.042318, -0.276441, -0.549585, -0.608116, -0.627627, -0.803219, -0.842240, -0.666647, -0.393503, -0.413013, -0.686157, -0.978812, -0.939791, -0.647137, -0.432524, -0.549585, -0.744688, -0.549585, -0.354482, -0.549585, -0.783709, -0.744688, -0.295952, 0.269847},
      {0.445440, 0.152785, -0.217910, -0.510565, -0.686157, -0.510565, -0.237421, -0.178890, -0.354482, -0.549585, -0.803219, -0.822729, -0.764199, -0.900771, -0.959301, -0.783709, -0.627627, -0.686157, -0.920281, -1.154404, -1.076363, -0.842240, -0.705668, -0.686157, -0.803219, -0.647137, -0.608116, -0.686157, -0.920281, -0.861750, -0.256931, 0.269847},
      {0.406419, 0.074744, -0.237421, -0.588606, -0.803219, -0.725178, -0.530075, -0.491054, -0.666647, -0.822729, -0.978812, -0.978812, -0.861750, -0.939791, -1.056853, -0.959301, -0.842240, -0.920281, -1.056853, -1.212935, -1.115384, -0.920281, -0.920281, -0.822729, -0.900771, -0.725178, -0.744688, -0.803219, -0.978812, -0.881260, -0.198400, 0.308868},
      {0.308868, -0.003297, -0.334972, -0.666647, -0.959301, -0.939791, -0.803219, -0.803219, -0.920281, -0.998322, -1.134894, -1.076363, -0.920281, -0.959301, -1.134894, -1.115384, -1.017832, -1.095874, -1.095874, -1.193425, -1.017832, -0.959301, -1.076363, -0.900771, -0.978812, -0.822729, -0.861750, -0.861750, -0.959301, -0.842240, -0.120359, 0.367398},
      {0.250337, -0.081338, -0.413013, -0.744688, -1.037343, -1.076363, -0.959301, -0.959301, -1.076363, -1.115384, -1.193425, -1.115384, -0.920281, -0.978812, -1.173915, -1.193425, -1.115384, -1.173915, -1.115384, -1.154404, -0.959301, -0.959301, -1.173915, -0.978812, -1.056853, -0.861750, -0.881260, -0.861750, -0.920281, -0.803219, -0.081338, 0.425929},
      {0.211316, -0.022808, -0.393503, -0.744688, -0.959301, -0.998322, -0.939791, -0.842240, -0.920281, -0.900771, -0.900771, -0.920281, -0.920281, -0.920281, -0.939791, -0.939791, -0.861750, -0.842240, -0.842240, -0.822729, -0.842240, -0.861750, -0.861750, -0.861750, -0.569096, -0.647137, -0.686157, -0.686157, -0.510565, -0.237421, 0.055234, 0.230826}},
     {{-0.587708, -0.906049, -0.846360, -0.925946, -0.687190, -0.508123, -0.468330, -0.528019, -0.428537, -0.508123, -0.647397, -0.647397, -0.707086, -0.766775, -0.726982, -0.547915, -0.508123, -0.488226, -0.587708, -0.547915, -0.368848, -0.368848, -0.428537, -0.408641, -0.408641, -0.528019, -0.687190, -0.567812, -0.030611, 0.526487, 0.725451, 0.864725},
      {-0.607604, -0.886153, -0.806568, -0.826464, -0.607604, -0.408641, -0.329056, -0.428537, -0.647397, -0.667293, -0.607604, -0.528019, -0.528019, -0.587708, -0.528019, -0.428537, -0.488226, -0.508123, -0.707086, -0.687190, -0.448434, -0.388745, -0.448434, -0.448434, -0.408641, -0.488226, -0.667293, -0.587708, -0.050507, 0.427005, 0.705554, 0.824932},
      {-0.687190, -0.886153, -0.806568, -0.786671, -0.547915, -0.289263, -0.070403, -0.269367, -0.687190, -0.567812, -0.289263, -0.070403, -0.050507, -0.050507, -0.010714, 0.009182, -0.229574, -0.348952, -0.607604, -0.587708, -0.309159, -0.169885, -0.289263, -0.269367, -0.388745, -0.468330, -0.647397, -0.627501, -0.110196, 0.327524, 0.645865, 0.824932},
      {-0.667293, -0.826464, -0.726982, -0.647397, -0.388745, -0.090300, 0.228042, -0.070403, -0.428537, -0.249470, 0.208146, 0.546384, 0.526487, 0.506591, 0.506591, 0.407109, 0.108664, -0.130092, -0.468330, -0.428537, -0.050507, 0.108664, -0.030611, -0.090300, -0.269367, -0.348952, -0.567812, -0.627501, -0.209678, 0.267835, 0.606072, 0.805036},
      {-0.627501, -0.786671, -0.647397, -0.508123, -0.269367, 0.088768, 0.566280, 0.148457, -0.289263, -0.070403, 0.486694, 0.824932, 0.785140, 0.824932, 0.844829, 0.725451, 0.247938, -0.070403, -0.448434, -0.408641, 0.088768, 0.267835, 0.068871, -0.090300, -0.110196, -0.189781, -0.448434, -0.587708, -0.269367, 0.208146, 0.566280, 0.805036},
      {-0.627501, -0.707086, -0.567812, -0.428537, -0.169885, 0.287731, 0.765243, 0.327524, -0.329056, -0.110196, 0.427005, 0.685658, 0.606072, 0.725451, 0.864725, 0.765243, 0.267835, -0.090300, -0.488226, -0.388745, 0.108664, 0.307627, 0.068871, -0.189781, 0.088768, 0.009182, -0.368848, -0.528019, -0.289263, 0.148457, 0.486694, 0.765243},
      {-0.587708, -0.647397, -0.547915, -0.348952, -0.090300, 0.367316, 0.944310, 0.427005, -0.189781, -0.050507, 0.347420, 0.486694, 0.287731, 0.446902, 0.625969, 0.526487, 0.307627, -0.070403, -0.528019, -0.408641, 0.168353, 0.367316, 0.068871, -0.229574, 0.228042, 0.088768, -0.289263, -0.488226, -0.289263, 0.168353, 0.486694, 0.765243},
      {-0.687190, -0.667293, -0.567812, -0.428537, -0.050507, 0.427005, 0.904518, 0.606072, -0.448434, -0.528019, -0.249470, -0.090300, -0.269367, -0.070403, 0.148457, 0.088768, -0.149989, -0.329056, -0.667293, -0.408641, 0.287731, 0.327524, -0.110196, -0.269367, 0.367316, 0.387213, -0.030611, -0.348952, -0.229574, 0.148457, 0.526487, 0.645865},
      {-0.607604, -0.647397, -0.607604, -0.448434, -0.090300, 0.367316, 0.864725, 0.606072, -0.269367, -0.488226, -0.388745, -0.348952, -0.607604, -0.428537, -0.169885, -0.289263, -0.428537, -0.528019, -0.786671, -0.428537, 0.287731, 0.327524, -0.070403, -0.229574, 0.347420, 0.347420, -0.070403, -0.388745, -0.209678, 0.148457, 0.526487, 0.665762},
      {-0.508123, -0.607604, -0.587708, -0.468330, -0.149989, 0.287731, 0.824932, 0.586176, -0.169885, -0.488226, -0.567812, -0.647397, -0.925946, -0.726982, -0.368848, -0.388745, -0.448434, -0.567812, -0.687190, -0.289263, 0.367316, 0.446902, 0.068871, -0.110196, 0.327524, 0.307627, -0.149989, -0.408641, -0.289263, 0.088768, 0.486694, 0.665762},
      {-0.408641, -0.547915, -0.607604, -0.528019, -0.209678, 0.208146, 0.745347, 0.586176, -0.070403, -0.547915, -0.726982, -0.806568, -1.025427, -0.766775, -0.408641, -0.408641, -0.428537, -0.587708, -0.726982, -0.448434, 0.128560, 0.208146, -0.030611, -0.149989, 0.267835, 0.247938, -0.229574, -0.508123, -0.329056, 0.048975, 0.486694, 0.685658},
      {-0.289263, -0.547915, -0.627501, -0.567812, -0.289263, 0.108664, 0.705554, 0.625969, -0.030611, -0.607604, -0.826464, -0.866257, -1.005531, -0.806568, -0.428537, -0.408641, -0.408641, -0.627501, -0.886153, -0.667293, -0.249470, -0.050507, -0.090300, -0.110196, 0.208146, 0.148457, -0.309159, -0.528019, -0.408641, -0.050507, 0.466798, 0.685658},
      {-0.269367, -0.587708, -0.707086, -0.587708, -0.289263, 0.088768, 0.705554, 0.705554, 0.048975, -0.587708, -0.846360, -0.746879, -0.766775, -0.468330, -0.070403, -0.050507, -0.289263, -0.687190, -0.965738, -0.826464, -0.448434, -0.209678, -0.070403, 0.009182, 0.148457, 0.029079, -0.428537, -0.607604, -0.428537, -0.110196, 0.407109, 0.725451},
      {-0.249470, -0.607604, -0.766775, -0.647397, -0.329056, 0.048975, 0.705554, 0.765243, 0.188249, -0.528019, -0.786671, -0.647397, -0.567812, -0.130092, 0.307627, 0.387213, -0.149989, -0.567812, -0.886153, -0.846360, -0.567812, -0.309159, -0.110196, 0.048975, 0.088768, 0.009182, -0.488226, -0.667293, -0.488226, -0.149989, 0.407109, 0.765243},
      {-0.249470, -0.607604, -0.786671, -0.627501, -0.309159, 0.048975, 0.725451, 0.785140, 0.307627, -0.448434, -0.766775, -0.707086, -0.607604, -0.149989, 0.307627, 0.367316, 0.188249, -0.249470, -0.587708, -0.587708, -0.408641, -0.169885, 0.048975, 0.208146, 0.068871, -0.070403, -0.547915, -0.667293, -0.468330, -0.149989, 0.387213, 0.765243},
      {-0.090300, -0.587708, -0.746879, -0.766775, -0.746879, -0.149989, 0.645865, 0.864725, 0.407109, -0.388745, -0.607604, -0.687190, -1.005531, -1.005531, -0.528019, 0.009182, 0.407109, 0.586176, 0.108664, -0.249470, -0.010714, 0.208146, 0.108664, -0.329056, -0.309159, -0.050507, 0.088768, -0.090300, -0.348952, -0.269367, 0.188249, 0.645865},
      {-0.090300, -0.567812, -0.746879, -0.766775, -0.746879, -0.130092, 0.606072, 0.805036, 0.188249, -0.547915, -0.667293, -0.687190, -1.005531, -1.065220, -0.726982, -0.209678, 0.307627, 0.546384, 0.208146, 0.009182, 0.188249, 0.387213, 0.128560, -0.368848, -0.348952, -0.030611, 0.168353, 0.009182, -0.229574, -0.229574, 0.188249, 0.625969},
      {-0.090300, -0.528019, -0.726982, -0.766775, -0.746879, -0.149989, 0.546384, 0.745347, 0.188249, -0.448434, -0.448434, -0.388745, -0.647397, -0.826464, -0.687190, -0.309159, 0.148457, 0.427005, 0.247938, 0.088768, 0.267835, 0.387213, 0.088768, -0.448434, -0.408641, -0.030611, 0.267835, 0.188249, -0.110196, -0.169885, 0.188249, 0.606072},
      {-0.050507, -0.448434, -0.726982, -0.786671, -0.746879, -0.189781, 0.486694, 0.685658, 0.307627, -0.149989, -0.090300, 0.009182, -0.269367, -0.547915, -0.587708, -0.309159, 0.188249, 0.486694, 0.387213, 0.208146, 0.327524, 0.446902, 0.228042, -0.348952, -0.488226, -0.030611, 0.347420, 0.327524, 0.009182, -0.070403, 0.188249, 0.546384},
      {-0.050507, -0.428537, -0.687190, -0.806568, -0.746879, -0.209678, 0.367316, 0.586176, 0.287731, -0.070403, 0.029079, 0.168353, -0.110196, -0.448434, -0.647397, -0.448434, 0.108664, 0.446902, 0.387213, 0.247938, 0.307627, 0.427005, 0.228042, -0.408641, -0.607604, -0.050507, 0.407109, 0.387213, 0.068871, -0.070403, 0.168353, 0.486694},
      {-0.030611, -0.368848, -0.707086, -0.826464, -0.746879, -0.289263, 0.307627, 0.526487, 0.287731, 0.009182, 0.108664, 0.148457, -0.070403, -0.428537, -0.707086, -0.547915, -0.070403, 0.287731, 0.327524, 0.267835, 0.247938, 0.307627, 0.088768, -0.547915, -0.687190, -0.090300, 0.407109, 0.387213, 0.009182, -0.149989, 0.088768, 0.446902},
      {-0.010714, -0.388745, -0.687190, -0.886153, -0.766775, -0.309159, 0.208146, 0.486694, 0.327524, 0.108664, 0.148457, 0.168353, -0.090300, -0.448434, -0.707086, -0.567812, 0.029079, 0.287731, 0.347420, 0.247938, 0.228042, 0.327524, 0.148457, -0.488226, -0.746879, -0.169885, 0.347420, 0.347420, -0.030611, -0.209678, 0.048975, 0.407109},
      {-0.030611, -0.368848, -0.766775, -0.886153, -0.766775, -0.329056, 0.168353, 0.466798, 0.287731, 0.108664, 0.088768, 0.068871, -0.130092, -0.488226, -0.746879, -0.587708, 0.068871, 0.267835, 0.228042, 0.068871, 0.009182, 0.188249, 0.148457, -0.428537, -0.786671, -0.209678, 0.327524, 0.327524, -0.070403, -0.289263, 0.009182, 0.367316},
      {0.048975, -0.249470, -0.528019, -0.687190, -0.687190, -0.348952, 0.148457, 0.407109, 0.347420, 0.108664, -0.090300, -0.010714, -0.110196, -0.508123, -0.647397, -0.428537, 0.029079, 0.148457, -0.070403, -0.289263, -0.368848, -0.030611, 0.208146, -0.189781, -0.647397, -0.309159, 0.208146, 0.108664, -0.249470, -0.428537, -0.229574, 0.287731},
      {0.068871, -0.189781, -0.468330, -0.627501, -0.667293, -0.348952, 0.148457, 0.367316, 0.267835, 0.048975, -0.149989, -0.070403, -0.189781, -0.508123, -0.587708, -0.408641, 0.029079, 0.068871, -0.149989, -0.448434, -0.508123, -0.169885, 0.088768, -0.229574, -0.607604, -0.348952, 0.088768, -0.010714, -0.388745, -0.488226, -0.269367, 0.267835},
      {0.128560, -0.130092, -0.428537, -0.587708, -0.667293, -0.368848, 0.068871, 0.247938, 0.128560, -0.070403, -0.289263, -0.229574, -0.249470, -0.528019, -0.627501, -0.448434, -0.110196, -0.110196, -0.408641, -0.707086, -0.726982, -0.388745, -0.110196, -0.309159, -0.607604, -0.388745, -0.110196, -0.229574, -0.567812, -0.627501, -0.309159, 0.228042},
      {0.188249, -0.090300, -0.329056, -0.528019, -0.667293, -0.428537, -0.110196, 0.048975, -0.110196, -0.249470, -0.468330, -0.408641, -0.388745, -0.587708, -0.687190, -0.508123, -0.309159, -0.368848, -0.687190, -0.985635, -0.945842, -0.627501, -0.368848, -0.408641, -0.607604, -0.468330, -0.309159, -0.468330, -0.806568, -0.806568, -0.348952, 0.188249},
      {0.208146, -0.030611, -0.368848, -0.587708, -0.746879, -0.627501, -0.348952, -0.229574, -0.368848, -0.508123, -0.687190, -0.627501, -0.528019, -0.667293, -0.786671, -0.627501, -0.508123, -0.647397, -0.925946, -1.164702, -1.085117, -0.786671, -0.627501, -0.547915, -0.667293, -0.567812, -0.547915, -0.667293, -0.925946, -0.906049, -0.368848, 0.208146},
      {0.128560, -0.149989, -0.388745, -0.647397, -0.886153, -0.826464, -0.627501, -0.547915, -0.667293, -0.766775, -0.906049, -0.826464, -0.667293, -0.746879, -0.886153, -0.806568, -0.766775, -0.886153, -1.065220, -1.224391, -1.124909, -0.886153, -0.886153, -0.746879, -0.826464, -0.687190, -0.687190, -0.786671, -0.985635, -0.925946, -0.309159, 0.247938},
      {0.048975, -0.209678, -0.468330, -0.766775, -1.025427, -1.025427, -0.886153, -0.846360, -0.965738, -1.005531, -1.085117, -0.965738, -0.766775, -0.826464, -1.005531, -1.045324, -0.945842, -1.085117, -1.105013, -1.184598, -1.025427, -0.906049, -1.045324, -0.866257, -0.945842, -0.786671, -0.846360, -0.846360, -0.965738, -0.906049, -0.229574, 0.247938},
      {0.029079, -0.289263, -0.547915, -0.826464, -1.144806, -1.144806, -1.025427, -1.025427, -1.105013, -1.144806, -1.204495, -1.065220, -0.886153, -0.906049, -1.105013, -1.124909, -1.105013, -1.164702, -1.164702, -1.184598, -0.945842, -0.886153, -1.124909, -0.925946, -1.045324, -0.906049, -0.925946, -0.886153, -0.985635, -0.886153, -0.189781, 0.307627},
      {-0.010714, -0.229574, -0.547915, -0.866257, -1.045324, -1.085117, -1.025427, -0.886153, -0.965738, -0.965738, -0.965738, -0.965738, -0.965738, -0.925946, -0.945842, -0.945842, -0.866257, -0.906049, -0.906049, -0.886153, -0.866257, -0.866257, -0.846360, -0.886153, -0.647397, -0.726982, -0.786671, -0.766775, -0.587708, -0.348952, -0.090300, 0.048975}}}};