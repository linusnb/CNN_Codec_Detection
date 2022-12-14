?	??s@??s@!??s@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??s@?i?WV?c@1	Q????a@A{?Fw;??I??qn.@*	/?$f??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?ݑ??? @!????`?X@)m;m?? @1n????X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?v?$$Ү?!?@????)?v?$$Ү?1?@????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?wG?j???!?????)?wG?j???1?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismn?2d???!??ĺ??)?>;??b??1??Ǜc??:Preprocessing2F
Iterator::Model???5????!7=?????)??֦??v?1?o4?4???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??
U1?J@Q

???G@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?i?WV?c@?i?WV?c@!?i?WV?c@      ??!       "		Q????a@	Q????a@!	Q????a@*      ??!       2	{?Fw;??{?Fw;??!{?Fw;??:	??qn.@??qn.@!??qn.@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??
U1?J@y

???G@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltervz?d???!vz?d???0"j
?gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropInputConv2DBackpropInput!!?V??!??~h׏??0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput??$?И??!*??6??0"=
sequential/conv_layer2/Relu_FusedConv2D?3?R&h??!?8P??"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?? ??]??!)KH????0"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGradY3????!?|???G??"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?*?????!?.??????0"-
IteratorGetNext/_2_Recv3p$E????!?u3?Yd??"K
-gradient_tape/sequential/conv_layer1/ReluGradReluGradݦL
DƧ?!_@?4????"k
Agradient_tape/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3??Q??!x^???8??Q      Y@Y????E@a?cp>?L@qM???d?8@y??DAk?s?"?

both?Your program is POTENTIALLY input-bound because 51.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?24.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 