?	?R?r???@?R?r???@!?R?r???@	?WNO???WNO??!?WNO??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?R?r???@?G?3?d@1??????@A??$?pt??I???,>@Y??4F?(??*	T㥛???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2? Q0c$@!????5?X@)$???E?#@1=??єX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@???߾??!??'eı??)???߾??1??'eı??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?O?eo??!?A?-e??)?O?eo??1?A?-e??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??ْU??!M??) ?&?f??1ً2j??:Preprocessing2F
Iterator::Model?M?g\??!4?9?????)??O?t?1?rI?؂??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?WNO??I??m`?? @Q5j?"??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?G?3?d@?G?3?d@!?G?3?d@      ??!       "	??????@??????@!??????@*      ??!       2	??$?pt????$?pt??!??$?pt??:	???,>@???,>@!???,>@B      ??!       J	??4F?(????4F?(??!??4F?(??R      ??!       Z	??4F?(????4F?(??!??4F?(??b      ??!       JGPUY?WNO??b q??m`?? @y5j?"??V@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterӌ?38???!ӌ?38???0"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter1??:.s??!lP+QϷ??0"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter`?P?)???!????l??0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?7?ho!??!?.ރ???0"=
sequential/conv_layer2/Relu_FusedConv2D?{?????!?c???h??"=
sequential/conv_layer4/Relu_FusedConv2D??r????!f?u?t???"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput?f??6???!??3???0"=
sequential/conv_layer3/Relu_FusedConv2D?5?읃??!4y?B???"j
?gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropInputConv2DBackpropInput"??$???!T?Cbt??0"j
?gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropInputConv2DBackpropInput"??!???!q?*?0h??0Q      Y@Y?i?C??-@a?R??
CU@qzf??2@y?t??:?"?

both?Your program is POTENTIALLY input-bound because 8.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?18.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 