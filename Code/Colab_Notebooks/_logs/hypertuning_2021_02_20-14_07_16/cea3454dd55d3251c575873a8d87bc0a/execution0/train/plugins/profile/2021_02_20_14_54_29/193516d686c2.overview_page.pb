?	???B??~@???B??~@!???B??~@	|*<v???|*<v???!|*<v???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???B??~@??"j??d@1c?T4?4t@A???~????I?_????@Y??~????*	??n?q?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2j/?혺"@!??6?=?X@)M???"@18?5?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?\?????!?V/C???)?\?????1?V/C???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchEկt><??!?L?P??)Eկt><??1?L?P??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismX??C???!J??u=??)?rg&Ε?1Q]-????:Preprocessing2F
Iterator::Model??o?N??!?$??!???)v?r??s?1f?i?:??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 33.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9|*<v???I??????A@Q	?CȕMP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??"j??d@??"j??d@!??"j??d@      ??!       "	c?T4?4t@c?T4?4t@!c?T4?4t@*      ??!       2	???~???????~????!???~????:	?_????@?_????@!?_????@B      ??!       J	??~??????~????!??~????R      ??!       Z	??~??????~????!??~????b      ??!       JGPUY|*<v???b q??????A@y	?CȕMP@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterap<?4??!ap<?4??0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput)?$?,??!{9E?=Y??0"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???o2??!??=W????0"=
sequential/conv_layer2/Relu_FusedConv2D'9y?ӷ?!?@?5~???"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??D??ԧ?!9m???G??0"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGrad???????!?{E????"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput??O촤?!1F?	????0"K
-gradient_tape/sequential/conv_layer1/ReluGradReluGrad.;?;????!?Y?M???"=
sequential/conv_layer3/Relu_FusedConv2D?H6????!*$???"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltern?dVD??!?0G?????0Q      Y@Yio?,?@@aKH?i?P@q`?zWs?&@y?*?t?"_?"?

both?Your program is POTENTIALLY input-bound because 33.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?11.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 