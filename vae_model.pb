
?
XPlaceholder*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
?
YPlaceholder*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
2
	keep_probPlaceholder*
dtype0*
shape: 
B
Reshape/shapeConst*
dtype0*
valueB"ÿÿÿÿ  
;
ReshapeReshapeYReshape/shape*
T0*
Tshape0
R
encoder/Reshape/shapeConst*%
valueB"ÿÿÿÿ         *
dtype0
K
encoder/ReshapeReshapeXencoder/Reshape/shape*
T0*
Tshape0

6encoder/conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"         @   *(
_class
loc:@encoder/conv2d/kernel*
dtype0

4encoder/conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *½*(
_class
loc:@encoder/conv2d/kernel*
dtype0

4encoder/conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *=*(
_class
loc:@encoder/conv2d/kernel*
dtype0
à
>encoder/conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform6encoder/conv2d/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@encoder/conv2d/kernel*
dtype0*
seed2 *

seed 
Ú
4encoder/conv2d/kernel/Initializer/random_uniform/subSub4encoder/conv2d/kernel/Initializer/random_uniform/max4encoder/conv2d/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@encoder/conv2d/kernel
ä
4encoder/conv2d/kernel/Initializer/random_uniform/mulMul>encoder/conv2d/kernel/Initializer/random_uniform/RandomUniform4encoder/conv2d/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@encoder/conv2d/kernel
Ö
0encoder/conv2d/kernel/Initializer/random_uniformAdd4encoder/conv2d/kernel/Initializer/random_uniform/mul4encoder/conv2d/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@encoder/conv2d/kernel

encoder/conv2d/kernel
VariableV2*
shared_name *(
_class
loc:@encoder/conv2d/kernel*
dtype0*
	container *
shape:@
Ë
encoder/conv2d/kernel/AssignAssignencoder/conv2d/kernel0encoder/conv2d/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@encoder/conv2d/kernel*
validate_shape(
p
encoder/conv2d/kernel/readIdentityencoder/conv2d/kernel*
T0*(
_class
loc:@encoder/conv2d/kernel
~
%encoder/conv2d/bias/Initializer/zerosConst*
dtype0*
valueB@*    *&
_class
loc:@encoder/conv2d/bias

encoder/conv2d/bias
VariableV2*&
_class
loc:@encoder/conv2d/bias*
dtype0*
	container *
shape:@*
shared_name 
º
encoder/conv2d/bias/AssignAssignencoder/conv2d/bias%encoder/conv2d/bias/Initializer/zeros*&
_class
loc:@encoder/conv2d/bias*
validate_shape(*
use_locking(*
T0
j
encoder/conv2d/bias/readIdentityencoder/conv2d/bias*
T0*&
_class
loc:@encoder/conv2d/bias
Q
encoder/conv2d/dilation_rateConst*
valueB"      *
dtype0
¬
encoder/conv2d/Conv2DConv2Dencoder/Reshapeencoder/conv2d/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
T0
r
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2Dencoder/conv2d/bias/read*
data_formatNHWC*
T0
K
encoder/conv2d/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
d
encoder/conv2d/LeakyRelu/mulMulencoder/conv2d/LeakyRelu/alphaencoder/conv2d/BiasAdd*
T0
j
 encoder/conv2d/LeakyRelu/MaximumMaximumencoder/conv2d/LeakyRelu/mulencoder/conv2d/BiasAdd*
T0
Y
encoder/dropout/ShapeShape encoder/conv2d/LeakyRelu/Maximum*
T0*
out_type0
O
"encoder/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
O
"encoder/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0

,encoder/dropout/random_uniform/RandomUniformRandomUniformencoder/dropout/Shape*
dtype0*
seed2 *

seed *
T0
z
"encoder/dropout/random_uniform/subSub"encoder/dropout/random_uniform/max"encoder/dropout/random_uniform/min*
T0

"encoder/dropout/random_uniform/mulMul,encoder/dropout/random_uniform/RandomUniform"encoder/dropout/random_uniform/sub*
T0
v
encoder/dropout/random_uniformAdd"encoder/dropout/random_uniform/mul"encoder/dropout/random_uniform/min*
T0
N
encoder/dropout/addAdd	keep_probencoder/dropout/random_uniform*
T0
<
encoder/dropout/FloorFloorencoder/dropout/add*
T0
T
encoder/dropout/divRealDiv encoder/conv2d/LeakyRelu/Maximum	keep_prob*
T0
O
encoder/dropout/mulMulencoder/dropout/divencoder/dropout/Floor*
T0
¡
8encoder/conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   **
_class 
loc:@encoder/conv2d_1/kernel*
dtype0

6encoder/conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *×³]½**
_class 
loc:@encoder/conv2d_1/kernel*
dtype0

6encoder/conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *×³]=**
_class 
loc:@encoder/conv2d_1/kernel*
dtype0
æ
@encoder/conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform8encoder/conv2d_1/kernel/Initializer/random_uniform/shape**
_class 
loc:@encoder/conv2d_1/kernel*
dtype0*
seed2 *

seed *
T0
â
6encoder/conv2d_1/kernel/Initializer/random_uniform/subSub6encoder/conv2d_1/kernel/Initializer/random_uniform/max6encoder/conv2d_1/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@encoder/conv2d_1/kernel
ì
6encoder/conv2d_1/kernel/Initializer/random_uniform/mulMul@encoder/conv2d_1/kernel/Initializer/random_uniform/RandomUniform6encoder/conv2d_1/kernel/Initializer/random_uniform/sub**
_class 
loc:@encoder/conv2d_1/kernel*
T0
Þ
2encoder/conv2d_1/kernel/Initializer/random_uniformAdd6encoder/conv2d_1/kernel/Initializer/random_uniform/mul6encoder/conv2d_1/kernel/Initializer/random_uniform/min**
_class 
loc:@encoder/conv2d_1/kernel*
T0

encoder/conv2d_1/kernel
VariableV2*
shape:@@*
shared_name **
_class 
loc:@encoder/conv2d_1/kernel*
dtype0*
	container 
Ó
encoder/conv2d_1/kernel/AssignAssignencoder/conv2d_1/kernel2encoder/conv2d_1/kernel/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@encoder/conv2d_1/kernel*
validate_shape(
v
encoder/conv2d_1/kernel/readIdentityencoder/conv2d_1/kernel**
_class 
loc:@encoder/conv2d_1/kernel*
T0

'encoder/conv2d_1/bias/Initializer/zerosConst*
valueB@*    *(
_class
loc:@encoder/conv2d_1/bias*
dtype0

encoder/conv2d_1/bias
VariableV2*
shared_name *(
_class
loc:@encoder/conv2d_1/bias*
dtype0*
	container *
shape:@
Â
encoder/conv2d_1/bias/AssignAssignencoder/conv2d_1/bias'encoder/conv2d_1/bias/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@encoder/conv2d_1/bias*
validate_shape(
p
encoder/conv2d_1/bias/readIdentityencoder/conv2d_1/bias*
T0*(
_class
loc:@encoder/conv2d_1/bias
S
encoder/conv2d_2/dilation_rateConst*
valueB"      *
dtype0
´
encoder/conv2d_2/Conv2DConv2Dencoder/dropout/mulencoder/conv2d_1/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
x
encoder/conv2d_2/BiasAddBiasAddencoder/conv2d_2/Conv2Dencoder/conv2d_1/bias/read*
data_formatNHWC*
T0
M
 encoder/conv2d_2/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
j
encoder/conv2d_2/LeakyRelu/mulMul encoder/conv2d_2/LeakyRelu/alphaencoder/conv2d_2/BiasAdd*
T0
p
"encoder/conv2d_2/LeakyRelu/MaximumMaximumencoder/conv2d_2/LeakyRelu/mulencoder/conv2d_2/BiasAdd*
T0
]
encoder/dropout_1/ShapeShape"encoder/conv2d_2/LeakyRelu/Maximum*
T0*
out_type0
Q
$encoder/dropout_1/random_uniform/minConst*
valueB
 *    *
dtype0
Q
$encoder/dropout_1/random_uniform/maxConst*
valueB
 *  ?*
dtype0

.encoder/dropout_1/random_uniform/RandomUniformRandomUniformencoder/dropout_1/Shape*

seed *
T0*
dtype0*
seed2 

$encoder/dropout_1/random_uniform/subSub$encoder/dropout_1/random_uniform/max$encoder/dropout_1/random_uniform/min*
T0

$encoder/dropout_1/random_uniform/mulMul.encoder/dropout_1/random_uniform/RandomUniform$encoder/dropout_1/random_uniform/sub*
T0
|
 encoder/dropout_1/random_uniformAdd$encoder/dropout_1/random_uniform/mul$encoder/dropout_1/random_uniform/min*
T0
R
encoder/dropout_1/addAdd	keep_prob encoder/dropout_1/random_uniform*
T0
@
encoder/dropout_1/FloorFloorencoder/dropout_1/add*
T0
X
encoder/dropout_1/divRealDiv"encoder/conv2d_2/LeakyRelu/Maximum	keep_prob*
T0
U
encoder/dropout_1/mulMulencoder/dropout_1/divencoder/dropout_1/Floor*
T0
¡
8encoder/conv2d_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"      @   @   **
_class 
loc:@encoder/conv2d_2/kernel

6encoder/conv2d_2/kernel/Initializer/random_uniform/minConst*
valueB
 *×³]½**
_class 
loc:@encoder/conv2d_2/kernel*
dtype0

6encoder/conv2d_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *×³]=**
_class 
loc:@encoder/conv2d_2/kernel*
dtype0
æ
@encoder/conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform8encoder/conv2d_2/kernel/Initializer/random_uniform/shape*

seed *
T0**
_class 
loc:@encoder/conv2d_2/kernel*
dtype0*
seed2 
â
6encoder/conv2d_2/kernel/Initializer/random_uniform/subSub6encoder/conv2d_2/kernel/Initializer/random_uniform/max6encoder/conv2d_2/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@encoder/conv2d_2/kernel
ì
6encoder/conv2d_2/kernel/Initializer/random_uniform/mulMul@encoder/conv2d_2/kernel/Initializer/random_uniform/RandomUniform6encoder/conv2d_2/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@encoder/conv2d_2/kernel
Þ
2encoder/conv2d_2/kernel/Initializer/random_uniformAdd6encoder/conv2d_2/kernel/Initializer/random_uniform/mul6encoder/conv2d_2/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@encoder/conv2d_2/kernel

encoder/conv2d_2/kernel
VariableV2*
shape:@@*
shared_name **
_class 
loc:@encoder/conv2d_2/kernel*
dtype0*
	container 
Ó
encoder/conv2d_2/kernel/AssignAssignencoder/conv2d_2/kernel2encoder/conv2d_2/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0**
_class 
loc:@encoder/conv2d_2/kernel
v
encoder/conv2d_2/kernel/readIdentityencoder/conv2d_2/kernel*
T0**
_class 
loc:@encoder/conv2d_2/kernel

'encoder/conv2d_2/bias/Initializer/zerosConst*
valueB@*    *(
_class
loc:@encoder/conv2d_2/bias*
dtype0

encoder/conv2d_2/bias
VariableV2*
shape:@*
shared_name *(
_class
loc:@encoder/conv2d_2/bias*
dtype0*
	container 
Â
encoder/conv2d_2/bias/AssignAssignencoder/conv2d_2/bias'encoder/conv2d_2/bias/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@encoder/conv2d_2/bias*
validate_shape(
p
encoder/conv2d_2/bias/readIdentityencoder/conv2d_2/bias*
T0*(
_class
loc:@encoder/conv2d_2/bias
S
encoder/conv2d_3/dilation_rateConst*
valueB"      *
dtype0
¶
encoder/conv2d_3/Conv2DConv2Dencoder/dropout_1/mulencoder/conv2d_2/kernel/read*
paddingSAME*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
x
encoder/conv2d_3/BiasAddBiasAddencoder/conv2d_3/Conv2Dencoder/conv2d_2/bias/read*
data_formatNHWC*
T0
M
 encoder/conv2d_3/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
j
encoder/conv2d_3/LeakyRelu/mulMul encoder/conv2d_3/LeakyRelu/alphaencoder/conv2d_3/BiasAdd*
T0
p
"encoder/conv2d_3/LeakyRelu/MaximumMaximumencoder/conv2d_3/LeakyRelu/mulencoder/conv2d_3/BiasAdd*
T0
]
encoder/dropout_2/ShapeShape"encoder/conv2d_3/LeakyRelu/Maximum*
T0*
out_type0
Q
$encoder/dropout_2/random_uniform/minConst*
valueB
 *    *
dtype0
Q
$encoder/dropout_2/random_uniform/maxConst*
valueB
 *  ?*
dtype0

.encoder/dropout_2/random_uniform/RandomUniformRandomUniformencoder/dropout_2/Shape*

seed *
T0*
dtype0*
seed2 

$encoder/dropout_2/random_uniform/subSub$encoder/dropout_2/random_uniform/max$encoder/dropout_2/random_uniform/min*
T0

$encoder/dropout_2/random_uniform/mulMul.encoder/dropout_2/random_uniform/RandomUniform$encoder/dropout_2/random_uniform/sub*
T0
|
 encoder/dropout_2/random_uniformAdd$encoder/dropout_2/random_uniform/mul$encoder/dropout_2/random_uniform/min*
T0
R
encoder/dropout_2/addAdd	keep_prob encoder/dropout_2/random_uniform*
T0
@
encoder/dropout_2/FloorFloorencoder/dropout_2/add*
T0
X
encoder/dropout_2/divRealDiv"encoder/conv2d_3/LeakyRelu/Maximum	keep_prob*
T0
U
encoder/dropout_2/mulMulencoder/dropout_2/divencoder/dropout_2/Floor*
T0
V
encoder/Flatten/flatten/ShapeShapeencoder/dropout_2/mul*
out_type0*
T0
Y
+encoder/Flatten/flatten/strided_slice/stackConst*
valueB: *
dtype0
[
-encoder/Flatten/flatten/strided_slice/stack_1Const*
dtype0*
valueB:
[
-encoder/Flatten/flatten/strided_slice/stack_2Const*
dtype0*
valueB:
Ù
%encoder/Flatten/flatten/strided_sliceStridedSliceencoder/Flatten/flatten/Shape+encoder/Flatten/flatten/strided_slice/stack-encoder/Flatten/flatten/strided_slice/stack_1-encoder/Flatten/flatten/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
Z
'encoder/Flatten/flatten/Reshape/shape/1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0

%encoder/Flatten/flatten/Reshape/shapePack%encoder/Flatten/flatten/strided_slice'encoder/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N

encoder/Flatten/flatten/ReshapeReshapeencoder/dropout_2/mul%encoder/Flatten/flatten/Reshape/shape*
T0*
Tshape0

5encoder/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"@     *'
_class
loc:@encoder/dense/kernel*
dtype0

3encoder/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *Aï2½*'
_class
loc:@encoder/dense/kernel*
dtype0

3encoder/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *Aï2=*'
_class
loc:@encoder/dense/kernel*
dtype0
Ý
=encoder/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5encoder/dense/kernel/Initializer/random_uniform/shape*
T0*'
_class
loc:@encoder/dense/kernel*
dtype0*
seed2 *

seed 
Ö
3encoder/dense/kernel/Initializer/random_uniform/subSub3encoder/dense/kernel/Initializer/random_uniform/max3encoder/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@encoder/dense/kernel
à
3encoder/dense/kernel/Initializer/random_uniform/mulMul=encoder/dense/kernel/Initializer/random_uniform/RandomUniform3encoder/dense/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@encoder/dense/kernel
Ò
/encoder/dense/kernel/Initializer/random_uniformAdd3encoder/dense/kernel/Initializer/random_uniform/mul3encoder/dense/kernel/Initializer/random_uniform/min*'
_class
loc:@encoder/dense/kernel*
T0

encoder/dense/kernel
VariableV2*
	container *
shape:	À*
shared_name *'
_class
loc:@encoder/dense/kernel*
dtype0
Ç
encoder/dense/kernel/AssignAssignencoder/dense/kernel/encoder/dense/kernel/Initializer/random_uniform*
T0*'
_class
loc:@encoder/dense/kernel*
validate_shape(*
use_locking(
m
encoder/dense/kernel/readIdentityencoder/dense/kernel*'
_class
loc:@encoder/dense/kernel*
T0
|
$encoder/dense/bias/Initializer/zerosConst*
valueB*    *%
_class
loc:@encoder/dense/bias*
dtype0

encoder/dense/bias
VariableV2*%
_class
loc:@encoder/dense/bias*
dtype0*
	container *
shape:*
shared_name 
¶
encoder/dense/bias/AssignAssignencoder/dense/bias$encoder/dense/bias/Initializer/zeros*%
_class
loc:@encoder/dense/bias*
validate_shape(*
use_locking(*
T0
g
encoder/dense/bias/readIdentityencoder/dense/bias*
T0*%
_class
loc:@encoder/dense/bias

encoder/dense/MatMulMatMulencoder/Flatten/flatten/Reshapeencoder/dense/kernel/read*
transpose_a( *
transpose_b( *
T0
o
encoder/dense/BiasAddBiasAddencoder/dense/MatMulencoder/dense/bias/read*
T0*
data_formatNHWC

7encoder/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@     *)
_class
loc:@encoder/dense_1/kernel*
dtype0

5encoder/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *Aï2½*)
_class
loc:@encoder/dense_1/kernel*
dtype0

5encoder/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *Aï2=*)
_class
loc:@encoder/dense_1/kernel*
dtype0
ã
?encoder/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7encoder/dense_1/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@encoder/dense_1/kernel*
dtype0*
seed2 *

seed 
Þ
5encoder/dense_1/kernel/Initializer/random_uniform/subSub5encoder/dense_1/kernel/Initializer/random_uniform/max5encoder/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@encoder/dense_1/kernel
è
5encoder/dense_1/kernel/Initializer/random_uniform/mulMul?encoder/dense_1/kernel/Initializer/random_uniform/RandomUniform5encoder/dense_1/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@encoder/dense_1/kernel
Ú
1encoder/dense_1/kernel/Initializer/random_uniformAdd5encoder/dense_1/kernel/Initializer/random_uniform/mul5encoder/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@encoder/dense_1/kernel

encoder/dense_1/kernel
VariableV2*
shape:	À*
shared_name *)
_class
loc:@encoder/dense_1/kernel*
dtype0*
	container 
Ï
encoder/dense_1/kernel/AssignAssignencoder/dense_1/kernel1encoder/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@encoder/dense_1/kernel*
validate_shape(
s
encoder/dense_1/kernel/readIdentityencoder/dense_1/kernel*
T0*)
_class
loc:@encoder/dense_1/kernel

&encoder/dense_1/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@encoder/dense_1/bias*
dtype0

encoder/dense_1/bias
VariableV2*'
_class
loc:@encoder/dense_1/bias*
dtype0*
	container *
shape:*
shared_name 
¾
encoder/dense_1/bias/AssignAssignencoder/dense_1/bias&encoder/dense_1/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@encoder/dense_1/bias*
validate_shape(
m
encoder/dense_1/bias/readIdentityencoder/dense_1/bias*
T0*'
_class
loc:@encoder/dense_1/bias

encoder/dense_2/MatMulMatMulencoder/Flatten/flatten/Reshapeencoder/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b( 
u
encoder/dense_2/BiasAddBiasAddencoder/dense_2/MatMulencoder/dense_1/bias/read*
T0*
data_formatNHWC
:
encoder/mul/xConst*
valueB
 *   ?*
dtype0
C
encoder/mulMulencoder/mul/xencoder/dense_2/BiasAdd*
T0
P
encoder/ShapeShapeencoder/Flatten/flatten/Reshape*
T0*
out_type0
I
encoder/strided_slice/stackConst*
valueB: *
dtype0
K
encoder/strided_slice/stack_1Const*
valueB:*
dtype0
K
encoder/strided_slice/stack_2Const*
valueB:*
dtype0

encoder/strided_sliceStridedSliceencoder/Shapeencoder/strided_slice/stackencoder/strided_slice/stack_1encoder/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
9
encoder/stack/1Const*
value	B :*
dtype0
[
encoder/stackPackencoder/strided_sliceencoder/stack/1*
T0*

axis *
N
G
encoder/random_normal/meanConst*
valueB
 *    *
dtype0
I
encoder/random_normal/stddevConst*
valueB
 *  ?*
dtype0

*encoder/random_normal/RandomStandardNormalRandomStandardNormalencoder/stack*
seed2 *

seed *
T0*
dtype0
s
encoder/random_normal/mulMul*encoder/random_normal/RandomStandardNormalencoder/random_normal/stddev*
T0
\
encoder/random_normalAddencoder/random_normal/mulencoder/random_normal/mean*
T0
(
encoder/ExpExpencoder/mul*
T0
?
encoder/MulMulencoder/random_normalencoder/Exp*
T0
?
encoder/addAddencoder/dense/BiasAddencoder/Mul*
T0

5decoder/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *'
_class
loc:@decoder/dense/kernel*
dtype0

3decoder/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *×³Ý¾*'
_class
loc:@decoder/dense/kernel*
dtype0

3decoder/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *×³Ý>*'
_class
loc:@decoder/dense/kernel*
dtype0
Ý
=decoder/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5decoder/dense/kernel/Initializer/random_uniform/shape*
seed2 *

seed *
T0*'
_class
loc:@decoder/dense/kernel*
dtype0
Ö
3decoder/dense/kernel/Initializer/random_uniform/subSub3decoder/dense/kernel/Initializer/random_uniform/max3decoder/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@decoder/dense/kernel
à
3decoder/dense/kernel/Initializer/random_uniform/mulMul=decoder/dense/kernel/Initializer/random_uniform/RandomUniform3decoder/dense/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@decoder/dense/kernel
Ò
/decoder/dense/kernel/Initializer/random_uniformAdd3decoder/dense/kernel/Initializer/random_uniform/mul3decoder/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@decoder/dense/kernel

decoder/dense/kernel
VariableV2*
shape
:*
shared_name *'
_class
loc:@decoder/dense/kernel*
dtype0*
	container 
Ç
decoder/dense/kernel/AssignAssigndecoder/dense/kernel/decoder/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@decoder/dense/kernel*
validate_shape(
m
decoder/dense/kernel/readIdentitydecoder/dense/kernel*
T0*'
_class
loc:@decoder/dense/kernel
|
$decoder/dense/bias/Initializer/zerosConst*
dtype0*
valueB*    *%
_class
loc:@decoder/dense/bias

decoder/dense/bias
VariableV2*
	container *
shape:*
shared_name *%
_class
loc:@decoder/dense/bias*
dtype0
¶
decoder/dense/bias/AssignAssigndecoder/dense/bias$decoder/dense/bias/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@decoder/dense/bias*
validate_shape(
g
decoder/dense/bias/readIdentitydecoder/dense/bias*
T0*%
_class
loc:@decoder/dense/bias
u
decoder/dense/MatMulMatMulencoder/adddecoder/dense/kernel/read*
transpose_a( *
transpose_b( *
T0
o
decoder/dense/BiasAddBiasAdddecoder/dense/MatMuldecoder/dense/bias/read*
T0*
data_formatNHWC
J
decoder/dense/LeakyRelu/alphaConst*
valueB
 *ÍÌL>*
dtype0
a
decoder/dense/LeakyRelu/mulMuldecoder/dense/LeakyRelu/alphadecoder/dense/BiasAdd*
T0
g
decoder/dense/LeakyRelu/MaximumMaximumdecoder/dense/LeakyRelu/muldecoder/dense/BiasAdd*
T0

7decoder/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   1   *)
_class
loc:@decoder/dense_1/kernel*
dtype0

5decoder/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *-É¾*)
_class
loc:@decoder/dense_1/kernel*
dtype0

5decoder/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *-É>*)
_class
loc:@decoder/dense_1/kernel*
dtype0
ã
?decoder/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7decoder/dense_1/kernel/Initializer/random_uniform/shape*

seed *
T0*)
_class
loc:@decoder/dense_1/kernel*
dtype0*
seed2 
Þ
5decoder/dense_1/kernel/Initializer/random_uniform/subSub5decoder/dense_1/kernel/Initializer/random_uniform/max5decoder/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@decoder/dense_1/kernel
è
5decoder/dense_1/kernel/Initializer/random_uniform/mulMul?decoder/dense_1/kernel/Initializer/random_uniform/RandomUniform5decoder/dense_1/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@decoder/dense_1/kernel
Ú
1decoder/dense_1/kernel/Initializer/random_uniformAdd5decoder/dense_1/kernel/Initializer/random_uniform/mul5decoder/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@decoder/dense_1/kernel

decoder/dense_1/kernel
VariableV2*
	container *
shape
:1*
shared_name *)
_class
loc:@decoder/dense_1/kernel*
dtype0
Ï
decoder/dense_1/kernel/AssignAssigndecoder/dense_1/kernel1decoder/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@decoder/dense_1/kernel*
validate_shape(
s
decoder/dense_1/kernel/readIdentitydecoder/dense_1/kernel*
T0*)
_class
loc:@decoder/dense_1/kernel

&decoder/dense_1/bias/Initializer/zerosConst*
valueB1*    *'
_class
loc:@decoder/dense_1/bias*
dtype0

decoder/dense_1/bias
VariableV2*
shape:1*
shared_name *'
_class
loc:@decoder/dense_1/bias*
dtype0*
	container 
¾
decoder/dense_1/bias/AssignAssigndecoder/dense_1/bias&decoder/dense_1/bias/Initializer/zeros*
T0*'
_class
loc:@decoder/dense_1/bias*
validate_shape(*
use_locking(
m
decoder/dense_1/bias/readIdentitydecoder/dense_1/bias*
T0*'
_class
loc:@decoder/dense_1/bias

decoder/dense_2/MatMulMatMuldecoder/dense/LeakyRelu/Maximumdecoder/dense_1/kernel/read*
transpose_a( *
transpose_b( *
T0
u
decoder/dense_2/BiasAddBiasAdddecoder/dense_2/MatMuldecoder/dense_1/bias/read*
T0*
data_formatNHWC
L
decoder/dense_2/LeakyRelu/alphaConst*
dtype0*
valueB
 *ÍÌL>
g
decoder/dense_2/LeakyRelu/mulMuldecoder/dense_2/LeakyRelu/alphadecoder/dense_2/BiasAdd*
T0
m
!decoder/dense_2/LeakyRelu/MaximumMaximumdecoder/dense_2/LeakyRelu/muldecoder/dense_2/BiasAdd*
T0
R
decoder/Reshape/shapeConst*%
valueB"ÿÿÿÿ         *
dtype0
k
decoder/ReshapeReshape!decoder/dense_2/LeakyRelu/Maximumdecoder/Reshape/shape*
T0*
Tshape0
±
@decoder/conv2d_transpose/kernel/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"      @      *2
_class(
&$loc:@decoder/conv2d_transpose/kernel

>decoder/conv2d_transpose/kernel/Initializer/random_uniform/minConst*
valueB
 *½*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0

>decoder/conv2d_transpose/kernel/Initializer/random_uniform/maxConst*
valueB
 *=*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0
þ
Hdecoder/conv2d_transpose/kernel/Initializer/random_uniform/RandomUniformRandomUniform@decoder/conv2d_transpose/kernel/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0*
seed2 *

seed 

>decoder/conv2d_transpose/kernel/Initializer/random_uniform/subSub>decoder/conv2d_transpose/kernel/Initializer/random_uniform/max>decoder/conv2d_transpose/kernel/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel

>decoder/conv2d_transpose/kernel/Initializer/random_uniform/mulMulHdecoder/conv2d_transpose/kernel/Initializer/random_uniform/RandomUniform>decoder/conv2d_transpose/kernel/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel
þ
:decoder/conv2d_transpose/kernel/Initializer/random_uniformAdd>decoder/conv2d_transpose/kernel/Initializer/random_uniform/mul>decoder/conv2d_transpose/kernel/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel
¯
decoder/conv2d_transpose/kernel
VariableV2*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0*
	container *
shape:@*
shared_name 
ó
&decoder/conv2d_transpose/kernel/AssignAssigndecoder/conv2d_transpose/kernel:decoder/conv2d_transpose/kernel/Initializer/random_uniform*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
validate_shape(*
use_locking(*
T0

$decoder/conv2d_transpose/kernel/readIdentitydecoder/conv2d_transpose/kernel*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel

/decoder/conv2d_transpose/bias/Initializer/zerosConst*
valueB@*    *0
_class&
$"loc:@decoder/conv2d_transpose/bias*
dtype0

decoder/conv2d_transpose/bias
VariableV2*
shape:@*
shared_name *0
_class&
$"loc:@decoder/conv2d_transpose/bias*
dtype0*
	container 
â
$decoder/conv2d_transpose/bias/AssignAssigndecoder/conv2d_transpose/bias/decoder/conv2d_transpose/bias/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
validate_shape(

"decoder/conv2d_transpose/bias/readIdentitydecoder/conv2d_transpose/bias*
T0*0
_class&
$"loc:@decoder/conv2d_transpose/bias
Q
decoder/conv2d_transpose/ShapeShapedecoder/Reshape*
T0*
out_type0
Z
,decoder/conv2d_transpose/strided_slice/stackConst*
valueB: *
dtype0
\
.decoder/conv2d_transpose/strided_slice/stack_1Const*
dtype0*
valueB:
\
.decoder/conv2d_transpose/strided_slice/stack_2Const*
valueB:*
dtype0
Þ
&decoder/conv2d_transpose/strided_sliceStridedSlicedecoder/conv2d_transpose/Shape,decoder/conv2d_transpose/strided_slice/stack.decoder/conv2d_transpose/strided_slice/stack_1.decoder/conv2d_transpose/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
\
.decoder/conv2d_transpose/strided_slice_1/stackConst*
valueB:*
dtype0
^
0decoder/conv2d_transpose/strided_slice_1/stack_1Const*
valueB:*
dtype0
^
0decoder/conv2d_transpose/strided_slice_1/stack_2Const*
valueB:*
dtype0
æ
(decoder/conv2d_transpose/strided_slice_1StridedSlicedecoder/conv2d_transpose/Shape.decoder/conv2d_transpose/strided_slice_1/stack0decoder/conv2d_transpose/strided_slice_1/stack_10decoder/conv2d_transpose/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
\
.decoder/conv2d_transpose/strided_slice_2/stackConst*
valueB:*
dtype0
^
0decoder/conv2d_transpose/strided_slice_2/stack_1Const*
dtype0*
valueB:
^
0decoder/conv2d_transpose/strided_slice_2/stack_2Const*
valueB:*
dtype0
æ
(decoder/conv2d_transpose/strided_slice_2StridedSlicedecoder/conv2d_transpose/Shape.decoder/conv2d_transpose/strided_slice_2/stack0decoder/conv2d_transpose/strided_slice_2/stack_10decoder/conv2d_transpose/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
H
decoder/conv2d_transpose/mul/yConst*
dtype0*
value	B :
v
decoder/conv2d_transpose/mulMul(decoder/conv2d_transpose/strided_slice_1decoder/conv2d_transpose/mul/y*
T0
J
 decoder/conv2d_transpose/mul_1/yConst*
value	B :*
dtype0
z
decoder/conv2d_transpose/mul_1Mul(decoder/conv2d_transpose/strided_slice_2 decoder/conv2d_transpose/mul_1/y*
T0
J
 decoder/conv2d_transpose/stack/3Const*
value	B :@*
dtype0
Ì
decoder/conv2d_transpose/stackPack&decoder/conv2d_transpose/strided_slicedecoder/conv2d_transpose/muldecoder/conv2d_transpose/mul_1 decoder/conv2d_transpose/stack/3*
T0*

axis *
N
ù
+decoder/conv2d_transpose/conv2d_transpose_2Conv2DBackpropInputdecoder/conv2d_transpose/stack$decoder/conv2d_transpose/kernel/readdecoder/Reshape*
paddingSAME*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

 decoder/conv2d_transpose/BiasAddBiasAdd+decoder/conv2d_transpose/conv2d_transpose_2"decoder/conv2d_transpose/bias/read*
T0*
data_formatNHWC
P
decoder/conv2d_transpose/ReluRelu decoder/conv2d_transpose/BiasAdd*
T0
V
decoder/dropout/ShapeShapedecoder/conv2d_transpose/Relu*
T0*
out_type0
O
"decoder/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
O
"decoder/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0

,decoder/dropout/random_uniform/RandomUniformRandomUniformdecoder/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
z
"decoder/dropout/random_uniform/subSub"decoder/dropout/random_uniform/max"decoder/dropout/random_uniform/min*
T0

"decoder/dropout/random_uniform/mulMul,decoder/dropout/random_uniform/RandomUniform"decoder/dropout/random_uniform/sub*
T0
v
decoder/dropout/random_uniformAdd"decoder/dropout/random_uniform/mul"decoder/dropout/random_uniform/min*
T0
N
decoder/dropout/addAdd	keep_probdecoder/dropout/random_uniform*
T0
<
decoder/dropout/FloorFloordecoder/dropout/add*
T0
Q
decoder/dropout/divRealDivdecoder/conv2d_transpose/Relu	keep_prob*
T0
O
decoder/dropout/mulMuldecoder/dropout/divdecoder/dropout/Floor*
T0
µ
Bdecoder/conv2d_transpose_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0
£
@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/minConst*
valueB
 *×³]½*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0
£
@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *×³]=*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0

Jdecoder/conv2d_transpose_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformBdecoder/conv2d_transpose_1/kernel/Initializer/random_uniform/shape*

seed *
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0*
seed2 

@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/subSub@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/max@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel

@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/mulMulJdecoder/conv2d_transpose_1/kernel/Initializer/random_uniform/RandomUniform@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/sub*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
T0

<decoder/conv2d_transpose_1/kernel/Initializer/random_uniformAdd@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/mul@decoder/conv2d_transpose_1/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel
³
!decoder/conv2d_transpose_1/kernel
VariableV2*
shared_name *4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0*
	container *
shape:@@
û
(decoder/conv2d_transpose_1/kernel/AssignAssign!decoder/conv2d_transpose_1/kernel<decoder/conv2d_transpose_1/kernel/Initializer/random_uniform*
use_locking(*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
validate_shape(

&decoder/conv2d_transpose_1/kernel/readIdentity!decoder/conv2d_transpose_1/kernel*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel

1decoder/conv2d_transpose_1/bias/Initializer/zerosConst*
dtype0*
valueB@*    *2
_class(
&$loc:@decoder/conv2d_transpose_1/bias
£
decoder/conv2d_transpose_1/bias
VariableV2*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
dtype0*
	container *
shape:@*
shared_name 
ê
&decoder/conv2d_transpose_1/bias/AssignAssigndecoder/conv2d_transpose_1/bias1decoder/conv2d_transpose_1/bias/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
validate_shape(

$decoder/conv2d_transpose_1/bias/readIdentitydecoder/conv2d_transpose_1/bias*
T0*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias
W
 decoder/conv2d_transpose_2/ShapeShapedecoder/dropout/mul*
out_type0*
T0
\
.decoder/conv2d_transpose_2/strided_slice/stackConst*
valueB: *
dtype0
^
0decoder/conv2d_transpose_2/strided_slice/stack_1Const*
valueB:*
dtype0
^
0decoder/conv2d_transpose_2/strided_slice/stack_2Const*
dtype0*
valueB:
è
(decoder/conv2d_transpose_2/strided_sliceStridedSlice decoder/conv2d_transpose_2/Shape.decoder/conv2d_transpose_2/strided_slice/stack0decoder/conv2d_transpose_2/strided_slice/stack_10decoder/conv2d_transpose_2/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
^
0decoder/conv2d_transpose_2/strided_slice_1/stackConst*
valueB:*
dtype0
`
2decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
dtype0*
valueB:
`
2decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
valueB:*
dtype0
ð
*decoder/conv2d_transpose_2/strided_slice_1StridedSlice decoder/conv2d_transpose_2/Shape0decoder/conv2d_transpose_2/strided_slice_1/stack2decoder/conv2d_transpose_2/strided_slice_1/stack_12decoder/conv2d_transpose_2/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
^
0decoder/conv2d_transpose_2/strided_slice_2/stackConst*
valueB:*
dtype0
`
2decoder/conv2d_transpose_2/strided_slice_2/stack_1Const*
valueB:*
dtype0
`
2decoder/conv2d_transpose_2/strided_slice_2/stack_2Const*
valueB:*
dtype0
ð
*decoder/conv2d_transpose_2/strided_slice_2StridedSlice decoder/conv2d_transpose_2/Shape0decoder/conv2d_transpose_2/strided_slice_2/stack2decoder/conv2d_transpose_2/strided_slice_2/stack_12decoder/conv2d_transpose_2/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
J
 decoder/conv2d_transpose_2/mul/yConst*
value	B :*
dtype0
|
decoder/conv2d_transpose_2/mulMul*decoder/conv2d_transpose_2/strided_slice_1 decoder/conv2d_transpose_2/mul/y*
T0
L
"decoder/conv2d_transpose_2/mul_1/yConst*
value	B :*
dtype0

 decoder/conv2d_transpose_2/mul_1Mul*decoder/conv2d_transpose_2/strided_slice_2"decoder/conv2d_transpose_2/mul_1/y*
T0
L
"decoder/conv2d_transpose_2/stack/3Const*
value	B :@*
dtype0
Ö
 decoder/conv2d_transpose_2/stackPack(decoder/conv2d_transpose_2/strided_slicedecoder/conv2d_transpose_2/mul decoder/conv2d_transpose_2/mul_1"decoder/conv2d_transpose_2/stack/3*
N*
T0*

axis 

+decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput decoder/conv2d_transpose_2/stack&decoder/conv2d_transpose_1/kernel/readdecoder/dropout/mul*
paddingSAME*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
 
"decoder/conv2d_transpose_2/BiasAddBiasAdd+decoder/conv2d_transpose_2/conv2d_transpose$decoder/conv2d_transpose_1/bias/read*
data_formatNHWC*
T0
T
decoder/conv2d_transpose_2/ReluRelu"decoder/conv2d_transpose_2/BiasAdd*
T0
Z
decoder/dropout_1/ShapeShapedecoder/conv2d_transpose_2/Relu*
out_type0*
T0
Q
$decoder/dropout_1/random_uniform/minConst*
valueB
 *    *
dtype0
Q
$decoder/dropout_1/random_uniform/maxConst*
valueB
 *  ?*
dtype0

.decoder/dropout_1/random_uniform/RandomUniformRandomUniformdecoder/dropout_1/Shape*
T0*
dtype0*
seed2 *

seed 

$decoder/dropout_1/random_uniform/subSub$decoder/dropout_1/random_uniform/max$decoder/dropout_1/random_uniform/min*
T0

$decoder/dropout_1/random_uniform/mulMul.decoder/dropout_1/random_uniform/RandomUniform$decoder/dropout_1/random_uniform/sub*
T0
|
 decoder/dropout_1/random_uniformAdd$decoder/dropout_1/random_uniform/mul$decoder/dropout_1/random_uniform/min*
T0
R
decoder/dropout_1/addAdd	keep_prob decoder/dropout_1/random_uniform*
T0
@
decoder/dropout_1/FloorFloordecoder/dropout_1/add*
T0
U
decoder/dropout_1/divRealDivdecoder/conv2d_transpose_2/Relu	keep_prob*
T0
U
decoder/dropout_1/mulMuldecoder/dropout_1/divdecoder/dropout_1/Floor*
T0
µ
Bdecoder/conv2d_transpose_2/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0
£
@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/minConst*
valueB
 *×³]½*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0
£
@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *×³]=*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0

Jdecoder/conv2d_transpose_2/kernel/Initializer/random_uniform/RandomUniformRandomUniformBdecoder/conv2d_transpose_2/kernel/Initializer/random_uniform/shape*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0*
seed2 *

seed 

@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/subSub@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/max@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel

@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/mulMulJdecoder/conv2d_transpose_2/kernel/Initializer/random_uniform/RandomUniform@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/sub*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel

<decoder/conv2d_transpose_2/kernel/Initializer/random_uniformAdd@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/mul@decoder/conv2d_transpose_2/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel
³
!decoder/conv2d_transpose_2/kernel
VariableV2*
shared_name *4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0*
	container *
shape:@@
û
(decoder/conv2d_transpose_2/kernel/AssignAssign!decoder/conv2d_transpose_2/kernel<decoder/conv2d_transpose_2/kernel/Initializer/random_uniform*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
validate_shape(*
use_locking(*
T0

&decoder/conv2d_transpose_2/kernel/readIdentity!decoder/conv2d_transpose_2/kernel*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel

1decoder/conv2d_transpose_2/bias/Initializer/zerosConst*
valueB@*    *2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
dtype0
£
decoder/conv2d_transpose_2/bias
VariableV2*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
dtype0*
	container *
shape:@*
shared_name 
ê
&decoder/conv2d_transpose_2/bias/AssignAssigndecoder/conv2d_transpose_2/bias1decoder/conv2d_transpose_2/bias/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
validate_shape(

$decoder/conv2d_transpose_2/bias/readIdentitydecoder/conv2d_transpose_2/bias*
T0*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias
Y
 decoder/conv2d_transpose_3/ShapeShapedecoder/dropout_1/mul*
T0*
out_type0
\
.decoder/conv2d_transpose_3/strided_slice/stackConst*
valueB: *
dtype0
^
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
dtype0*
valueB:
^
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
valueB:*
dtype0
è
(decoder/conv2d_transpose_3/strided_sliceStridedSlice decoder/conv2d_transpose_3/Shape.decoder/conv2d_transpose_3/strided_slice/stack0decoder/conv2d_transpose_3/strided_slice/stack_10decoder/conv2d_transpose_3/strided_slice/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
^
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
valueB:*
dtype0
`
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
valueB:*
dtype0
`
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
valueB:*
dtype0
ð
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice decoder/conv2d_transpose_3/Shape0decoder/conv2d_transpose_3/strided_slice_1/stack2decoder/conv2d_transpose_3/strided_slice_1/stack_12decoder/conv2d_transpose_3/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
^
0decoder/conv2d_transpose_3/strided_slice_2/stackConst*
valueB:*
dtype0
`
2decoder/conv2d_transpose_3/strided_slice_2/stack_1Const*
valueB:*
dtype0
`
2decoder/conv2d_transpose_3/strided_slice_2/stack_2Const*
valueB:*
dtype0
ð
*decoder/conv2d_transpose_3/strided_slice_2StridedSlice decoder/conv2d_transpose_3/Shape0decoder/conv2d_transpose_3/strided_slice_2/stack2decoder/conv2d_transpose_3/strided_slice_2/stack_12decoder/conv2d_transpose_3/strided_slice_2/stack_2*
end_mask *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
J
 decoder/conv2d_transpose_3/mul/yConst*
value	B :*
dtype0
|
decoder/conv2d_transpose_3/mulMul*decoder/conv2d_transpose_3/strided_slice_1 decoder/conv2d_transpose_3/mul/y*
T0
L
"decoder/conv2d_transpose_3/mul_1/yConst*
value	B :*
dtype0

 decoder/conv2d_transpose_3/mul_1Mul*decoder/conv2d_transpose_3/strided_slice_2"decoder/conv2d_transpose_3/mul_1/y*
T0
L
"decoder/conv2d_transpose_3/stack/3Const*
value	B :@*
dtype0
Ö
 decoder/conv2d_transpose_3/stackPack(decoder/conv2d_transpose_3/strided_slicedecoder/conv2d_transpose_3/mul decoder/conv2d_transpose_3/mul_1"decoder/conv2d_transpose_3/stack/3*
T0*

axis *
N

+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput decoder/conv2d_transpose_3/stack&decoder/conv2d_transpose_2/kernel/readdecoder/dropout_1/mul*
paddingSAME*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
 
"decoder/conv2d_transpose_3/BiasAddBiasAdd+decoder/conv2d_transpose_3/conv2d_transpose$decoder/conv2d_transpose_2/bias/read*
T0*
data_formatNHWC
T
decoder/conv2d_transpose_3/ReluRelu"decoder/conv2d_transpose_3/BiasAdd*
T0
`
decoder/Flatten/flatten/ShapeShapedecoder/conv2d_transpose_3/Relu*
T0*
out_type0
Y
+decoder/Flatten/flatten/strided_slice/stackConst*
dtype0*
valueB: 
[
-decoder/Flatten/flatten/strided_slice/stack_1Const*
dtype0*
valueB:
[
-decoder/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0
Ù
%decoder/Flatten/flatten/strided_sliceStridedSlicedecoder/Flatten/flatten/Shape+decoder/Flatten/flatten/strided_slice/stack-decoder/Flatten/flatten/strided_slice/stack_1-decoder/Flatten/flatten/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
Z
'decoder/Flatten/flatten/Reshape/shape/1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0

%decoder/Flatten/flatten/Reshape/shapePack%decoder/Flatten/flatten/strided_slice'decoder/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N

decoder/Flatten/flatten/ReshapeReshapedecoder/conv2d_transpose_3/Relu%decoder/Flatten/flatten/Reshape/shape*
T0*
Tshape0

7decoder/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB" 1    *)
_class
loc:@decoder/dense_2/kernel*
dtype0

5decoder/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *6Ð­¼*)
_class
loc:@decoder/dense_2/kernel*
dtype0

5decoder/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *6Ð­<*)
_class
loc:@decoder/dense_2/kernel*
dtype0
ã
?decoder/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7decoder/dense_2/kernel/Initializer/random_uniform/shape*

seed *
T0*)
_class
loc:@decoder/dense_2/kernel*
dtype0*
seed2 
Þ
5decoder/dense_2/kernel/Initializer/random_uniform/subSub5decoder/dense_2/kernel/Initializer/random_uniform/max5decoder/dense_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@decoder/dense_2/kernel
è
5decoder/dense_2/kernel/Initializer/random_uniform/mulMul?decoder/dense_2/kernel/Initializer/random_uniform/RandomUniform5decoder/dense_2/kernel/Initializer/random_uniform/sub*)
_class
loc:@decoder/dense_2/kernel*
T0
Ú
1decoder/dense_2/kernel/Initializer/random_uniformAdd5decoder/dense_2/kernel/Initializer/random_uniform/mul5decoder/dense_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@decoder/dense_2/kernel

decoder/dense_2/kernel
VariableV2*
shared_name *)
_class
loc:@decoder/dense_2/kernel*
dtype0*
	container *
shape:
b
Ï
decoder/dense_2/kernel/AssignAssigndecoder/dense_2/kernel1decoder/dense_2/kernel/Initializer/random_uniform*
T0*)
_class
loc:@decoder/dense_2/kernel*
validate_shape(*
use_locking(
s
decoder/dense_2/kernel/readIdentitydecoder/dense_2/kernel*)
_class
loc:@decoder/dense_2/kernel*
T0

&decoder/dense_2/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@decoder/dense_2/bias*
dtype0

decoder/dense_2/bias
VariableV2*'
_class
loc:@decoder/dense_2/bias*
dtype0*
	container *
shape:*
shared_name 
¾
decoder/dense_2/bias/AssignAssigndecoder/dense_2/bias&decoder/dense_2/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@decoder/dense_2/bias*
validate_shape(
m
decoder/dense_2/bias/readIdentitydecoder/dense_2/bias*'
_class
loc:@decoder/dense_2/bias*
T0

decoder/dense_3/MatMulMatMuldecoder/Flatten/flatten/Reshapedecoder/dense_2/kernel/read*
T0*
transpose_a( *
transpose_b( 
u
decoder/dense_3/BiasAddBiasAdddecoder/dense_3/MatMuldecoder/dense_2/bias/read*
data_formatNHWC*
T0
D
decoder/dense_3/SigmoidSigmoiddecoder/dense_3/BiasAdd*
T0
P
decoder/Reshape_1/shapeConst*!
valueB"ÿÿÿÿ      *
dtype0
e
decoder/Reshape_1Reshapedecoder/dense_3/Sigmoiddecoder/Reshape_1/shape*
T0*
Tshape0
D
Reshape_1/shapeConst*
valueB"ÿÿÿÿ  *
dtype0
O
	Reshape_1Reshapedecoder/Reshape_1Reshape_1/shape*
T0*
Tshape0
C
SquaredDifferenceSquaredDifference	Reshape_1Reshape*
T0
?
Sum/reduction_indicesConst*
value	B :*
dtype0
Z
SumSumSquaredDifferenceSum/reduction_indices*
T0*
	keep_dims( *

Tidx0
2
mul/xConst*
valueB
 *   @*
dtype0
'
mulMulmul/xencoder/mul*
T0
2
add/xConst*
valueB
 *  ?*
dtype0

addAddadd/xmul*
T0
0
SquareSquareencoder/dense/BiasAdd*
T0
 
subSubaddSquare*
T0
4
mul_1/xConst*
dtype0*
valueB
 *   @
+
mul_1Mulmul_1/xencoder/mul*
T0

ExpExpmul_1*
T0

sub_1SubsubExp*
T0
A
Sum_1/reduction_indicesConst*
value	B :*
dtype0
R
Sum_1Sumsub_1Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0
4
mul_2/xConst*
dtype0*
valueB
 *   ¿
%
mul_2Mulmul_2/xSum_1*
T0
!
add_1AddSummul_2*
T0
3
ConstConst*
valueB: *
dtype0
@
MeanMeanadd_1Const*
	keep_dims( *

Tidx0*
T0
8
gradients/ShapeConst*
valueB *
dtype0
<
gradients/ConstConst*
valueB
 *  ?*
dtype0
A
gradients/FillFillgradients/Shapegradients/Const*
T0
O
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0
p
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0
B
gradients/Mean_grad/ShapeShapeadd_1*
T0*
out_type0
s
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0
D
gradients/Mean_grad/Shape_1Shapeadd_1*
T0*
out_type0
D
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0
w
gradients/Mean_grad/ConstConst*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
dtype0
®
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
	keep_dims( *

Tidx0
y
gradients/Mean_grad/Const_1Const*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: 
²
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
	keep_dims( *

Tidx0*
T0
w
gradients/Mean_grad/Maximum/yConst*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
value	B :*
dtype0

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
V
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0
c
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0
A
gradients/add_1_grad/ShapeShapeSum*
T0*
out_type0
E
gradients/add_1_grad/Shape_1Shapemul_2*
T0*
out_type0

*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0

gradients/add_1_grad/SumSumgradients/Mean_grad/truediv*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
t
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0

gradients/add_1_grad/Sum_1Sumgradients/Mean_grad/truediv,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
z
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
¹
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape
¿
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1
M
gradients/Sum_grad/ShapeShapeSquaredDifference*
T0*
out_type0
n
gradients/Sum_grad/SizeConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0

gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape

gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
p
gradients/Sum_grad/Shape_1Const*+
_class!
loc:@gradients/Sum_grad/Shape*
valueB *
dtype0
u
gradients/Sum_grad/range/startConst*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B : 
u
gradients/Sum_grad/range/deltaConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
³
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*+
_class!
loc:@gradients/Sum_grad/Shape*

Tidx0
t
gradients/Sum_grad/Fill/valueConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0

gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
Õ
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
N*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
s
gradients/Sum_grad/Maximum/yConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0

gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape

gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape

gradients/Sum_grad/ReshapeReshape-gradients/add_1_grad/tuple/control_dependency gradients/Sum_grad/DynamicStitch*
T0*
Tshape0
s
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0
C
gradients/mul_2_grad/ShapeConst*
dtype0*
valueB 
E
gradients/mul_2_grad/Shape_1ShapeSum_1*
T0*
out_type0

*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0
`
gradients/mul_2_grad/mulMul/gradients/add_1_grad/tuple/control_dependency_1Sum_1*
T0

gradients/mul_2_grad/SumSumgradients/mul_2_grad/mul*gradients/mul_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
t
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
Tshape0*
T0
d
gradients/mul_2_grad/mul_1Mulmul_2/x/gradients/add_1_grad/tuple/control_dependency_1*
T0

gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
z
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
T0*
Tshape0
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
¹
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape
¿
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1
S
&gradients/SquaredDifference_grad/ShapeShape	Reshape_1*
T0*
out_type0
S
(gradients/SquaredDifference_grad/Shape_1ShapeReshape*
T0*
out_type0
ª
6gradients/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/SquaredDifference_grad/Shape(gradients/SquaredDifference_grad/Shape_1*
T0
n
'gradients/SquaredDifference_grad/scalarConst^gradients/Sum_grad/Tile*
valueB
 *   @*
dtype0
v
$gradients/SquaredDifference_grad/mulMul'gradients/SquaredDifference_grad/scalargradients/Sum_grad/Tile*
T0
b
$gradients/SquaredDifference_grad/subSub	Reshape_1Reshape^gradients/Sum_grad/Tile*
T0

&gradients/SquaredDifference_grad/mul_1Mul$gradients/SquaredDifference_grad/mul$gradients/SquaredDifference_grad/sub*
T0
±
$gradients/SquaredDifference_grad/SumSum&gradients/SquaredDifference_grad/mul_16gradients/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0

(gradients/SquaredDifference_grad/ReshapeReshape$gradients/SquaredDifference_grad/Sum&gradients/SquaredDifference_grad/Shape*
Tshape0*
T0
µ
&gradients/SquaredDifference_grad/Sum_1Sum&gradients/SquaredDifference_grad/mul_18gradients/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0

*gradients/SquaredDifference_grad/Reshape_1Reshape&gradients/SquaredDifference_grad/Sum_1(gradients/SquaredDifference_grad/Shape_1*
T0*
Tshape0
`
$gradients/SquaredDifference_grad/NegNeg*gradients/SquaredDifference_grad/Reshape_1*
T0

1gradients/SquaredDifference_grad/tuple/group_depsNoOp)^gradients/SquaredDifference_grad/Reshape%^gradients/SquaredDifference_grad/Neg
é
9gradients/SquaredDifference_grad/tuple/control_dependencyIdentity(gradients/SquaredDifference_grad/Reshape2^gradients/SquaredDifference_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/SquaredDifference_grad/Reshape
ã
;gradients/SquaredDifference_grad/tuple/control_dependency_1Identity$gradients/SquaredDifference_grad/Neg2^gradients/SquaredDifference_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/SquaredDifference_grad/Neg
C
gradients/Sum_1_grad/ShapeShapesub_1*
T0*
out_type0
r
gradients/Sum_1_grad/SizeConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0

gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape

gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
T0
t
gradients/Sum_1_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
valueB *
dtype0
y
 gradients/Sum_1_grad/range/startConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B : *
dtype0
y
 gradients/Sum_1_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0
½
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*-
_class#
!loc:@gradients/Sum_1_grad/Shape*

Tidx0
x
gradients/Sum_1_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0

gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape
á
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
N
w
gradients/Sum_1_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0
£
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape

gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape

gradients/Sum_1_grad/ReshapeReshape/gradients/mul_2_grad/tuple/control_dependency_1"gradients/Sum_1_grad/DynamicStitch*
T0*
Tshape0
y
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*

Tmultiples0*
T0
S
gradients/Reshape_1_grad/ShapeShapedecoder/Reshape_1*
T0*
out_type0

 gradients/Reshape_1_grad/ReshapeReshape9gradients/SquaredDifference_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0
A
gradients/sub_1_grad/ShapeShapesub*
T0*
out_type0
C
gradients/sub_1_grad/Shape_1ShapeExp*
T0*
out_type0

*gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_1_grad/Shapegradients/sub_1_grad/Shape_1*
T0

gradients/sub_1_grad/SumSumgradients/Sum_1_grad/Tile*gradients/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
t
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
T0*
Tshape0

gradients/sub_1_grad/Sum_1Sumgradients/Sum_1_grad/Tile,gradients/sub_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
D
gradients/sub_1_grad/NegNeggradients/sub_1_grad/Sum_1*
T0
x
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Neggradients/sub_1_grad/Shape_1*
T0*
Tshape0
m
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/sub_1_grad/Reshape^gradients/sub_1_grad/Reshape_1
¹
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape
¿
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1*
T0
a
&gradients/decoder/Reshape_1_grad/ShapeShapedecoder/dense_3/Sigmoid*
T0*
out_type0

(gradients/decoder/Reshape_1_grad/ReshapeReshape gradients/Reshape_1_grad/Reshape&gradients/decoder/Reshape_1_grad/Shape*
T0*
Tshape0
?
gradients/sub_grad/ShapeShapeadd*
out_type0*
T0
D
gradients/sub_grad/Shape_1ShapeSquare*
T0*
out_type0

(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0

gradients/sub_grad/SumSum-gradients/sub_1_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
n
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0
 
gradients/sub_grad/Sum_1Sum-gradients/sub_1_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
@
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0
r
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
±
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
·
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
\
gradients/Exp_grad/mulMul/gradients/sub_1_grad/tuple/control_dependency_1Exp*
T0

2gradients/decoder/dense_3/Sigmoid_grad/SigmoidGradSigmoidGraddecoder/dense_3/Sigmoid(gradients/decoder/Reshape_1_grad/Reshape*
T0
A
gradients/add_grad/ShapeConst*
dtype0*
valueB 
A
gradients/add_grad/Shape_1Shapemul*
T0*
out_type0

(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0

gradients/add_grad/SumSum+gradients/sub_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
n
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0

gradients/add_grad/Sum_1Sum+gradients/sub_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
t
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
±
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
·
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
x
gradients/Square_grad/mul/xConst.^gradients/sub_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0
]
gradients/Square_grad/mulMulgradients/Square_grad/mul/xencoder/dense/BiasAdd*
T0
u
gradients/Square_grad/mul_1Mul-gradients/sub_grad/tuple/control_dependency_1gradients/Square_grad/mul*
T0
C
gradients/mul_1_grad/ShapeConst*
valueB *
dtype0
K
gradients/mul_1_grad/Shape_1Shapeencoder/mul*
T0*
out_type0

*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
T0
M
gradients/mul_1_grad/mulMulgradients/Exp_grad/mulencoder/mul*
T0

gradients/mul_1_grad/SumSumgradients/mul_1_grad/mul*gradients/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
t
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
T0*
Tshape0
K
gradients/mul_1_grad/mul_1Mulmul_1/xgradients/Exp_grad/mul*
T0

gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
z
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
T0*
Tshape0
m
%gradients/mul_1_grad/tuple/group_depsNoOp^gradients/mul_1_grad/Reshape^gradients/mul_1_grad/Reshape_1
¹
-gradients/mul_1_grad/tuple/control_dependencyIdentitygradients/mul_1_grad/Reshape&^gradients/mul_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_1_grad/Reshape
¿
/gradients/mul_1_grad/tuple/control_dependency_1Identitygradients/mul_1_grad/Reshape_1&^gradients/mul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1

2gradients/decoder/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients/decoder/dense_3/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC
©
7gradients/decoder/dense_3/BiasAdd_grad/tuple/group_depsNoOp3^gradients/decoder/dense_3/Sigmoid_grad/SigmoidGrad3^gradients/decoder/dense_3/BiasAdd_grad/BiasAddGrad

?gradients/decoder/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity2gradients/decoder/dense_3/Sigmoid_grad/SigmoidGrad8^gradients/decoder/dense_3/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/decoder/dense_3/Sigmoid_grad/SigmoidGrad

Agradients/decoder/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/decoder/dense_3/BiasAdd_grad/BiasAddGrad8^gradients/decoder/dense_3/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/decoder/dense_3/BiasAdd_grad/BiasAddGrad
A
gradients/mul_grad/ShapeConst*
valueB *
dtype0
I
gradients/mul_grad/Shape_1Shapeencoder/mul*
T0*
out_type0

(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0
b
gradients/mul_grad/mulMul-gradients/add_grad/tuple/control_dependency_1encoder/mul*
T0

gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
n
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0
^
gradients/mul_grad/mul_1Mulmul/x-gradients/add_grad/tuple/control_dependency_1*
T0

gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
t
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
±
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape
·
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
T0
Ã
,gradients/decoder/dense_3/MatMul_grad/MatMulMatMul?gradients/decoder/dense_3/BiasAdd_grad/tuple/control_dependencydecoder/dense_2/kernel/read*
transpose_b(*
T0*
transpose_a( 
É
.gradients/decoder/dense_3/MatMul_grad/MatMul_1MatMuldecoder/Flatten/flatten/Reshape?gradients/decoder/dense_3/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0

6gradients/decoder/dense_3/MatMul_grad/tuple/group_depsNoOp-^gradients/decoder/dense_3/MatMul_grad/MatMul/^gradients/decoder/dense_3/MatMul_grad/MatMul_1
û
>gradients/decoder/dense_3/MatMul_grad/tuple/control_dependencyIdentity,gradients/decoder/dense_3/MatMul_grad/MatMul7^gradients/decoder/dense_3/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/decoder/dense_3/MatMul_grad/MatMul

@gradients/decoder/dense_3/MatMul_grad/tuple/control_dependency_1Identity.gradients/decoder/dense_3/MatMul_grad/MatMul_17^gradients/decoder/dense_3/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients/decoder/dense_3/MatMul_grad/MatMul_1*
T0
w
4gradients/decoder/Flatten/flatten/Reshape_grad/ShapeShapedecoder/conv2d_transpose_3/Relu*
T0*
out_type0
Î
6gradients/decoder/Flatten/flatten/Reshape_grad/ReshapeReshape>gradients/decoder/dense_3/MatMul_grad/tuple/control_dependency4gradients/decoder/Flatten/flatten/Reshape_grad/Shape*
Tshape0*
T0
¥
7gradients/decoder/conv2d_transpose_3/Relu_grad/ReluGradReluGrad6gradients/decoder/Flatten/flatten/Reshape_grad/Reshapedecoder/conv2d_transpose_3/Relu*
T0
¥
=gradients/decoder/conv2d_transpose_3/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients/decoder/conv2d_transpose_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC
Ä
Bgradients/decoder/conv2d_transpose_3/BiasAdd_grad/tuple/group_depsNoOp8^gradients/decoder/conv2d_transpose_3/Relu_grad/ReluGrad>^gradients/decoder/conv2d_transpose_3/BiasAdd_grad/BiasAddGrad
©
Jgradients/decoder/conv2d_transpose_3/BiasAdd_grad/tuple/control_dependencyIdentity7gradients/decoder/conv2d_transpose_3/Relu_grad/ReluGradC^gradients/decoder/conv2d_transpose_3/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/decoder/conv2d_transpose_3/Relu_grad/ReluGrad
·
Lgradients/decoder/conv2d_transpose_3/BiasAdd_grad/tuple/control_dependency_1Identity=gradients/decoder/conv2d_transpose_3/BiasAdd_grad/BiasAddGradC^gradients/decoder/conv2d_transpose_3/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/decoder/conv2d_transpose_3/BiasAdd_grad/BiasAddGrad
}
@gradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/ShapeConst*%
valueB"      @   @   *
dtype0
ì
Ogradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilterJgradients/decoder/conv2d_transpose_3/BiasAdd_grad/tuple/control_dependency@gradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/Shapedecoder/dropout_1/mul*
paddingSAME*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

Agradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/Conv2DConv2DJgradients/decoder/conv2d_transpose_3/BiasAdd_grad/tuple/control_dependency&decoder/conv2d_transpose_2/kernel/read*
paddingSAME*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
é
Kgradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/tuple/group_depsNoOpP^gradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/Conv2DBackpropFilterB^gradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/Conv2D
ë
Sgradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/tuple/control_dependencyIdentityOgradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/Conv2DBackpropFilterL^gradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/Conv2DBackpropFilter
Ñ
Ugradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/tuple/control_dependency_1IdentityAgradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/Conv2DL^gradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/Conv2D
c
*gradients/decoder/dropout_1/mul_grad/ShapeShapedecoder/dropout_1/div*
T0*
out_type0
g
,gradients/decoder/dropout_1/mul_grad/Shape_1Shapedecoder/dropout_1/Floor*
T0*
out_type0
¶
:gradients/decoder/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/decoder/dropout_1/mul_grad/Shape,gradients/decoder/dropout_1/mul_grad/Shape_1*
T0
¨
(gradients/decoder/dropout_1/mul_grad/mulMulUgradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/tuple/control_dependency_1decoder/dropout_1/Floor*
T0
»
(gradients/decoder/dropout_1/mul_grad/SumSum(gradients/decoder/dropout_1/mul_grad/mul:gradients/decoder/dropout_1/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
¤
,gradients/decoder/dropout_1/mul_grad/ReshapeReshape(gradients/decoder/dropout_1/mul_grad/Sum*gradients/decoder/dropout_1/mul_grad/Shape*
Tshape0*
T0
¨
*gradients/decoder/dropout_1/mul_grad/mul_1Muldecoder/dropout_1/divUgradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/tuple/control_dependency_1*
T0
Á
*gradients/decoder/dropout_1/mul_grad/Sum_1Sum*gradients/decoder/dropout_1/mul_grad/mul_1<gradients/decoder/dropout_1/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
ª
.gradients/decoder/dropout_1/mul_grad/Reshape_1Reshape*gradients/decoder/dropout_1/mul_grad/Sum_1,gradients/decoder/dropout_1/mul_grad/Shape_1*
T0*
Tshape0

5gradients/decoder/dropout_1/mul_grad/tuple/group_depsNoOp-^gradients/decoder/dropout_1/mul_grad/Reshape/^gradients/decoder/dropout_1/mul_grad/Reshape_1
ù
=gradients/decoder/dropout_1/mul_grad/tuple/control_dependencyIdentity,gradients/decoder/dropout_1/mul_grad/Reshape6^gradients/decoder/dropout_1/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/decoder/dropout_1/mul_grad/Reshape
ÿ
?gradients/decoder/dropout_1/mul_grad/tuple/control_dependency_1Identity.gradients/decoder/dropout_1/mul_grad/Reshape_16^gradients/decoder/dropout_1/mul_grad/tuple/group_deps*A
_class7
53loc:@gradients/decoder/dropout_1/mul_grad/Reshape_1*
T0
m
*gradients/decoder/dropout_1/div_grad/ShapeShapedecoder/conv2d_transpose_2/Relu*
T0*
out_type0
U
,gradients/decoder/dropout_1/div_grad/Shape_1Const*
valueB *
dtype0
¶
:gradients/decoder/dropout_1/div_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/decoder/dropout_1/div_grad/Shape,gradients/decoder/dropout_1/div_grad/Shape_1*
T0

,gradients/decoder/dropout_1/div_grad/RealDivRealDiv=gradients/decoder/dropout_1/mul_grad/tuple/control_dependency	keep_prob*
T0
¿
(gradients/decoder/dropout_1/div_grad/SumSum,gradients/decoder/dropout_1/div_grad/RealDiv:gradients/decoder/dropout_1/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
¤
,gradients/decoder/dropout_1/div_grad/ReshapeReshape(gradients/decoder/dropout_1/div_grad/Sum*gradients/decoder/dropout_1/div_grad/Shape*
T0*
Tshape0
Y
(gradients/decoder/dropout_1/div_grad/NegNegdecoder/conv2d_transpose_2/Relu*
T0
w
.gradients/decoder/dropout_1/div_grad/RealDiv_1RealDiv(gradients/decoder/dropout_1/div_grad/Neg	keep_prob*
T0
}
.gradients/decoder/dropout_1/div_grad/RealDiv_2RealDiv.gradients/decoder/dropout_1/div_grad/RealDiv_1	keep_prob*
T0
§
(gradients/decoder/dropout_1/div_grad/mulMul=gradients/decoder/dropout_1/mul_grad/tuple/control_dependency.gradients/decoder/dropout_1/div_grad/RealDiv_2*
T0
¿
*gradients/decoder/dropout_1/div_grad/Sum_1Sum(gradients/decoder/dropout_1/div_grad/mul<gradients/decoder/dropout_1/div_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
ª
.gradients/decoder/dropout_1/div_grad/Reshape_1Reshape*gradients/decoder/dropout_1/div_grad/Sum_1,gradients/decoder/dropout_1/div_grad/Shape_1*
T0*
Tshape0

5gradients/decoder/dropout_1/div_grad/tuple/group_depsNoOp-^gradients/decoder/dropout_1/div_grad/Reshape/^gradients/decoder/dropout_1/div_grad/Reshape_1
ù
=gradients/decoder/dropout_1/div_grad/tuple/control_dependencyIdentity,gradients/decoder/dropout_1/div_grad/Reshape6^gradients/decoder/dropout_1/div_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/decoder/dropout_1/div_grad/Reshape
ÿ
?gradients/decoder/dropout_1/div_grad/tuple/control_dependency_1Identity.gradients/decoder/dropout_1/div_grad/Reshape_16^gradients/decoder/dropout_1/div_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/decoder/dropout_1/div_grad/Reshape_1
¬
7gradients/decoder/conv2d_transpose_2/Relu_grad/ReluGradReluGrad=gradients/decoder/dropout_1/div_grad/tuple/control_dependencydecoder/conv2d_transpose_2/Relu*
T0
¥
=gradients/decoder/conv2d_transpose_2/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients/decoder/conv2d_transpose_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC
Ä
Bgradients/decoder/conv2d_transpose_2/BiasAdd_grad/tuple/group_depsNoOp8^gradients/decoder/conv2d_transpose_2/Relu_grad/ReluGrad>^gradients/decoder/conv2d_transpose_2/BiasAdd_grad/BiasAddGrad
©
Jgradients/decoder/conv2d_transpose_2/BiasAdd_grad/tuple/control_dependencyIdentity7gradients/decoder/conv2d_transpose_2/Relu_grad/ReluGradC^gradients/decoder/conv2d_transpose_2/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/decoder/conv2d_transpose_2/Relu_grad/ReluGrad
·
Lgradients/decoder/conv2d_transpose_2/BiasAdd_grad/tuple/control_dependency_1Identity=gradients/decoder/conv2d_transpose_2/BiasAdd_grad/BiasAddGradC^gradients/decoder/conv2d_transpose_2/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/decoder/conv2d_transpose_2/BiasAdd_grad/BiasAddGrad
}
@gradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/ShapeConst*%
valueB"      @   @   *
dtype0
ê
Ogradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilterJgradients/decoder/conv2d_transpose_2/BiasAdd_grad/tuple/control_dependency@gradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/Shapedecoder/dropout/mul*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
T0

Agradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/Conv2DConv2DJgradients/decoder/conv2d_transpose_2/BiasAdd_grad/tuple/control_dependency&decoder/conv2d_transpose_1/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
é
Kgradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/tuple/group_depsNoOpP^gradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/Conv2DBackpropFilterB^gradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/Conv2D
ë
Sgradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/tuple/control_dependencyIdentityOgradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/Conv2DBackpropFilterL^gradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/Conv2DBackpropFilter
Ñ
Ugradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/tuple/control_dependency_1IdentityAgradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/Conv2DL^gradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/Conv2D
_
(gradients/decoder/dropout/mul_grad/ShapeShapedecoder/dropout/div*
out_type0*
T0
c
*gradients/decoder/dropout/mul_grad/Shape_1Shapedecoder/dropout/Floor*
T0*
out_type0
°
8gradients/decoder/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/decoder/dropout/mul_grad/Shape*gradients/decoder/dropout/mul_grad/Shape_1*
T0
¤
&gradients/decoder/dropout/mul_grad/mulMulUgradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/tuple/control_dependency_1decoder/dropout/Floor*
T0
µ
&gradients/decoder/dropout/mul_grad/SumSum&gradients/decoder/dropout/mul_grad/mul8gradients/decoder/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0

*gradients/decoder/dropout/mul_grad/ReshapeReshape&gradients/decoder/dropout/mul_grad/Sum(gradients/decoder/dropout/mul_grad/Shape*
T0*
Tshape0
¤
(gradients/decoder/dropout/mul_grad/mul_1Muldecoder/dropout/divUgradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/tuple/control_dependency_1*
T0
»
(gradients/decoder/dropout/mul_grad/Sum_1Sum(gradients/decoder/dropout/mul_grad/mul_1:gradients/decoder/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
¤
,gradients/decoder/dropout/mul_grad/Reshape_1Reshape(gradients/decoder/dropout/mul_grad/Sum_1*gradients/decoder/dropout/mul_grad/Shape_1*
T0*
Tshape0

3gradients/decoder/dropout/mul_grad/tuple/group_depsNoOp+^gradients/decoder/dropout/mul_grad/Reshape-^gradients/decoder/dropout/mul_grad/Reshape_1
ñ
;gradients/decoder/dropout/mul_grad/tuple/control_dependencyIdentity*gradients/decoder/dropout/mul_grad/Reshape4^gradients/decoder/dropout/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/decoder/dropout/mul_grad/Reshape
÷
=gradients/decoder/dropout/mul_grad/tuple/control_dependency_1Identity,gradients/decoder/dropout/mul_grad/Reshape_14^gradients/decoder/dropout/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/decoder/dropout/mul_grad/Reshape_1
i
(gradients/decoder/dropout/div_grad/ShapeShapedecoder/conv2d_transpose/Relu*
T0*
out_type0
S
*gradients/decoder/dropout/div_grad/Shape_1Const*
valueB *
dtype0
°
8gradients/decoder/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/decoder/dropout/div_grad/Shape*gradients/decoder/dropout/div_grad/Shape_1*
T0

*gradients/decoder/dropout/div_grad/RealDivRealDiv;gradients/decoder/dropout/mul_grad/tuple/control_dependency	keep_prob*
T0
¹
&gradients/decoder/dropout/div_grad/SumSum*gradients/decoder/dropout/div_grad/RealDiv8gradients/decoder/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0

*gradients/decoder/dropout/div_grad/ReshapeReshape&gradients/decoder/dropout/div_grad/Sum(gradients/decoder/dropout/div_grad/Shape*
T0*
Tshape0
U
&gradients/decoder/dropout/div_grad/NegNegdecoder/conv2d_transpose/Relu*
T0
s
,gradients/decoder/dropout/div_grad/RealDiv_1RealDiv&gradients/decoder/dropout/div_grad/Neg	keep_prob*
T0
y
,gradients/decoder/dropout/div_grad/RealDiv_2RealDiv,gradients/decoder/dropout/div_grad/RealDiv_1	keep_prob*
T0
¡
&gradients/decoder/dropout/div_grad/mulMul;gradients/decoder/dropout/mul_grad/tuple/control_dependency,gradients/decoder/dropout/div_grad/RealDiv_2*
T0
¹
(gradients/decoder/dropout/div_grad/Sum_1Sum&gradients/decoder/dropout/div_grad/mul:gradients/decoder/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
¤
,gradients/decoder/dropout/div_grad/Reshape_1Reshape(gradients/decoder/dropout/div_grad/Sum_1*gradients/decoder/dropout/div_grad/Shape_1*
T0*
Tshape0

3gradients/decoder/dropout/div_grad/tuple/group_depsNoOp+^gradients/decoder/dropout/div_grad/Reshape-^gradients/decoder/dropout/div_grad/Reshape_1
ñ
;gradients/decoder/dropout/div_grad/tuple/control_dependencyIdentity*gradients/decoder/dropout/div_grad/Reshape4^gradients/decoder/dropout/div_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/decoder/dropout/div_grad/Reshape
÷
=gradients/decoder/dropout/div_grad/tuple/control_dependency_1Identity,gradients/decoder/dropout/div_grad/Reshape_14^gradients/decoder/dropout/div_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/decoder/dropout/div_grad/Reshape_1
¦
5gradients/decoder/conv2d_transpose/Relu_grad/ReluGradReluGrad;gradients/decoder/dropout/div_grad/tuple/control_dependencydecoder/conv2d_transpose/Relu*
T0
¡
;gradients/decoder/conv2d_transpose/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/decoder/conv2d_transpose/Relu_grad/ReluGrad*
T0*
data_formatNHWC
¾
@gradients/decoder/conv2d_transpose/BiasAdd_grad/tuple/group_depsNoOp6^gradients/decoder/conv2d_transpose/Relu_grad/ReluGrad<^gradients/decoder/conv2d_transpose/BiasAdd_grad/BiasAddGrad
¡
Hgradients/decoder/conv2d_transpose/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/decoder/conv2d_transpose/Relu_grad/ReluGradA^gradients/decoder/conv2d_transpose/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/decoder/conv2d_transpose/Relu_grad/ReluGrad
¯
Jgradients/decoder/conv2d_transpose/BiasAdd_grad/tuple/control_dependency_1Identity;gradients/decoder/conv2d_transpose/BiasAdd_grad/BiasAddGradA^gradients/decoder/conv2d_transpose/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/decoder/conv2d_transpose/BiasAdd_grad/BiasAddGrad
}
@gradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/ShapeConst*%
valueB"      @      *
dtype0
ä
Ogradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/Conv2DBackpropFilterConv2DBackpropFilterHgradients/decoder/conv2d_transpose/BiasAdd_grad/tuple/control_dependency@gradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/Shapedecoder/Reshape*
paddingSAME*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

Agradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/Conv2DConv2DHgradients/decoder/conv2d_transpose/BiasAdd_grad/tuple/control_dependency$decoder/conv2d_transpose/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
é
Kgradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/tuple/group_depsNoOpP^gradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/Conv2DBackpropFilterB^gradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/Conv2D
ë
Sgradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/tuple/control_dependencyIdentityOgradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/Conv2DBackpropFilterL^gradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/Conv2DBackpropFilter
Ñ
Ugradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/tuple/control_dependency_1IdentityAgradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/Conv2DL^gradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/Conv2D
i
$gradients/decoder/Reshape_grad/ShapeShape!decoder/dense_2/LeakyRelu/Maximum*
T0*
out_type0
Å
&gradients/decoder/Reshape_grad/ReshapeReshapeUgradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/tuple/control_dependency_1$gradients/decoder/Reshape_grad/Shape*
T0*
Tshape0
w
6gradients/decoder/dense_2/LeakyRelu/Maximum_grad/ShapeShapedecoder/dense_2/LeakyRelu/mul*
T0*
out_type0
s
8gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Shape_1Shapedecoder/dense_2/BiasAdd*
T0*
out_type0

8gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Shape_2Shape&gradients/decoder/Reshape_grad/Reshape*
out_type0*
T0
i
<gradients/decoder/dense_2/LeakyRelu/Maximum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
¿
6gradients/decoder/dense_2/LeakyRelu/Maximum_grad/zerosFill8gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Shape_2<gradients/decoder/dense_2/LeakyRelu/Maximum_grad/zeros/Const*
T0

=gradients/decoder/dense_2/LeakyRelu/Maximum_grad/GreaterEqualGreaterEqualdecoder/dense_2/LeakyRelu/muldecoder/dense_2/BiasAdd*
T0
Ú
Fgradients/decoder/dense_2/LeakyRelu/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Shape8gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Shape_1*
T0
é
7gradients/decoder/dense_2/LeakyRelu/Maximum_grad/SelectSelect=gradients/decoder/dense_2/LeakyRelu/Maximum_grad/GreaterEqual&gradients/decoder/Reshape_grad/Reshape6gradients/decoder/dense_2/LeakyRelu/Maximum_grad/zeros*
T0
ë
9gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Select_1Select=gradients/decoder/dense_2/LeakyRelu/Maximum_grad/GreaterEqual6gradients/decoder/dense_2/LeakyRelu/Maximum_grad/zeros&gradients/decoder/Reshape_grad/Reshape*
T0
â
4gradients/decoder/dense_2/LeakyRelu/Maximum_grad/SumSum7gradients/decoder/dense_2/LeakyRelu/Maximum_grad/SelectFgradients/decoder/dense_2/LeakyRelu/Maximum_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
È
8gradients/decoder/dense_2/LeakyRelu/Maximum_grad/ReshapeReshape4gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Sum6gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Shape*
T0*
Tshape0
è
6gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Sum_1Sum9gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Select_1Hgradients/decoder/dense_2/LeakyRelu/Maximum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
Î
:gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Reshape_1Reshape6gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Sum_18gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Shape_1*
T0*
Tshape0
Á
Agradients/decoder/dense_2/LeakyRelu/Maximum_grad/tuple/group_depsNoOp9^gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Reshape;^gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Reshape_1
©
Igradients/decoder/dense_2/LeakyRelu/Maximum_grad/tuple/control_dependencyIdentity8gradients/decoder/dense_2/LeakyRelu/Maximum_grad/ReshapeB^gradients/decoder/dense_2/LeakyRelu/Maximum_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Reshape
¯
Kgradients/decoder/dense_2/LeakyRelu/Maximum_grad/tuple/control_dependency_1Identity:gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Reshape_1B^gradients/decoder/dense_2/LeakyRelu/Maximum_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Reshape_1
[
2gradients/decoder/dense_2/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
o
4gradients/decoder/dense_2/LeakyRelu/mul_grad/Shape_1Shapedecoder/dense_2/BiasAdd*
T0*
out_type0
Î
Bgradients/decoder/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/decoder/dense_2/LeakyRelu/mul_grad/Shape4gradients/decoder/dense_2/LeakyRelu/mul_grad/Shape_1*
T0
¤
0gradients/decoder/dense_2/LeakyRelu/mul_grad/mulMulIgradients/decoder/dense_2/LeakyRelu/Maximum_grad/tuple/control_dependencydecoder/dense_2/BiasAdd*
T0
Ó
0gradients/decoder/dense_2/LeakyRelu/mul_grad/SumSum0gradients/decoder/dense_2/LeakyRelu/mul_grad/mulBgradients/decoder/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
¼
4gradients/decoder/dense_2/LeakyRelu/mul_grad/ReshapeReshape0gradients/decoder/dense_2/LeakyRelu/mul_grad/Sum2gradients/decoder/dense_2/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
®
2gradients/decoder/dense_2/LeakyRelu/mul_grad/mul_1Muldecoder/dense_2/LeakyRelu/alphaIgradients/decoder/dense_2/LeakyRelu/Maximum_grad/tuple/control_dependency*
T0
Ù
2gradients/decoder/dense_2/LeakyRelu/mul_grad/Sum_1Sum2gradients/decoder/dense_2/LeakyRelu/mul_grad/mul_1Dgradients/decoder/dense_2/LeakyRelu/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
Â
6gradients/decoder/dense_2/LeakyRelu/mul_grad/Reshape_1Reshape2gradients/decoder/dense_2/LeakyRelu/mul_grad/Sum_14gradients/decoder/dense_2/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
µ
=gradients/decoder/dense_2/LeakyRelu/mul_grad/tuple/group_depsNoOp5^gradients/decoder/dense_2/LeakyRelu/mul_grad/Reshape7^gradients/decoder/dense_2/LeakyRelu/mul_grad/Reshape_1

Egradients/decoder/dense_2/LeakyRelu/mul_grad/tuple/control_dependencyIdentity4gradients/decoder/dense_2/LeakyRelu/mul_grad/Reshape>^gradients/decoder/dense_2/LeakyRelu/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/decoder/dense_2/LeakyRelu/mul_grad/Reshape

Ggradients/decoder/dense_2/LeakyRelu/mul_grad/tuple/control_dependency_1Identity6gradients/decoder/dense_2/LeakyRelu/mul_grad/Reshape_1>^gradients/decoder/dense_2/LeakyRelu/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/decoder/dense_2/LeakyRelu/mul_grad/Reshape_1

gradients/AddNAddNKgradients/decoder/dense_2/LeakyRelu/Maximum_grad/tuple/control_dependency_1Ggradients/decoder/dense_2/LeakyRelu/mul_grad/tuple/control_dependency_1*
T0*M
_classC
A?loc:@gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Reshape_1*
N
q
2gradients/decoder/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
data_formatNHWC*
T0

7gradients/decoder/dense_2/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN3^gradients/decoder/dense_2/BiasAdd_grad/BiasAddGrad
í
?gradients/decoder/dense_2/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN8^gradients/decoder/dense_2/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/decoder/dense_2/LeakyRelu/Maximum_grad/Reshape_1

Agradients/decoder/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/decoder/dense_2/BiasAdd_grad/BiasAddGrad8^gradients/decoder/dense_2/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/decoder/dense_2/BiasAdd_grad/BiasAddGrad
Ã
,gradients/decoder/dense_2/MatMul_grad/MatMulMatMul?gradients/decoder/dense_2/BiasAdd_grad/tuple/control_dependencydecoder/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b(
É
.gradients/decoder/dense_2/MatMul_grad/MatMul_1MatMuldecoder/dense/LeakyRelu/Maximum?gradients/decoder/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(

6gradients/decoder/dense_2/MatMul_grad/tuple/group_depsNoOp-^gradients/decoder/dense_2/MatMul_grad/MatMul/^gradients/decoder/dense_2/MatMul_grad/MatMul_1
û
>gradients/decoder/dense_2/MatMul_grad/tuple/control_dependencyIdentity,gradients/decoder/dense_2/MatMul_grad/MatMul7^gradients/decoder/dense_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/decoder/dense_2/MatMul_grad/MatMul

@gradients/decoder/dense_2/MatMul_grad/tuple/control_dependency_1Identity.gradients/decoder/dense_2/MatMul_grad/MatMul_17^gradients/decoder/dense_2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/decoder/dense_2/MatMul_grad/MatMul_1
s
4gradients/decoder/dense/LeakyRelu/Maximum_grad/ShapeShapedecoder/dense/LeakyRelu/mul*
T0*
out_type0
o
6gradients/decoder/dense/LeakyRelu/Maximum_grad/Shape_1Shapedecoder/dense/BiasAdd*
T0*
out_type0

6gradients/decoder/dense/LeakyRelu/Maximum_grad/Shape_2Shape>gradients/decoder/dense_2/MatMul_grad/tuple/control_dependency*
T0*
out_type0
g
:gradients/decoder/dense/LeakyRelu/Maximum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
¹
4gradients/decoder/dense/LeakyRelu/Maximum_grad/zerosFill6gradients/decoder/dense/LeakyRelu/Maximum_grad/Shape_2:gradients/decoder/dense/LeakyRelu/Maximum_grad/zeros/Const*
T0

;gradients/decoder/dense/LeakyRelu/Maximum_grad/GreaterEqualGreaterEqualdecoder/dense/LeakyRelu/muldecoder/dense/BiasAdd*
T0
Ô
Dgradients/decoder/dense/LeakyRelu/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/decoder/dense/LeakyRelu/Maximum_grad/Shape6gradients/decoder/dense/LeakyRelu/Maximum_grad/Shape_1*
T0
û
5gradients/decoder/dense/LeakyRelu/Maximum_grad/SelectSelect;gradients/decoder/dense/LeakyRelu/Maximum_grad/GreaterEqual>gradients/decoder/dense_2/MatMul_grad/tuple/control_dependency4gradients/decoder/dense/LeakyRelu/Maximum_grad/zeros*
T0
ý
7gradients/decoder/dense/LeakyRelu/Maximum_grad/Select_1Select;gradients/decoder/dense/LeakyRelu/Maximum_grad/GreaterEqual4gradients/decoder/dense/LeakyRelu/Maximum_grad/zeros>gradients/decoder/dense_2/MatMul_grad/tuple/control_dependency*
T0
Ü
2gradients/decoder/dense/LeakyRelu/Maximum_grad/SumSum5gradients/decoder/dense/LeakyRelu/Maximum_grad/SelectDgradients/decoder/dense/LeakyRelu/Maximum_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
Â
6gradients/decoder/dense/LeakyRelu/Maximum_grad/ReshapeReshape2gradients/decoder/dense/LeakyRelu/Maximum_grad/Sum4gradients/decoder/dense/LeakyRelu/Maximum_grad/Shape*
T0*
Tshape0
â
4gradients/decoder/dense/LeakyRelu/Maximum_grad/Sum_1Sum7gradients/decoder/dense/LeakyRelu/Maximum_grad/Select_1Fgradients/decoder/dense/LeakyRelu/Maximum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
È
8gradients/decoder/dense/LeakyRelu/Maximum_grad/Reshape_1Reshape4gradients/decoder/dense/LeakyRelu/Maximum_grad/Sum_16gradients/decoder/dense/LeakyRelu/Maximum_grad/Shape_1*
Tshape0*
T0
»
?gradients/decoder/dense/LeakyRelu/Maximum_grad/tuple/group_depsNoOp7^gradients/decoder/dense/LeakyRelu/Maximum_grad/Reshape9^gradients/decoder/dense/LeakyRelu/Maximum_grad/Reshape_1
¡
Ggradients/decoder/dense/LeakyRelu/Maximum_grad/tuple/control_dependencyIdentity6gradients/decoder/dense/LeakyRelu/Maximum_grad/Reshape@^gradients/decoder/dense/LeakyRelu/Maximum_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/decoder/dense/LeakyRelu/Maximum_grad/Reshape
§
Igradients/decoder/dense/LeakyRelu/Maximum_grad/tuple/control_dependency_1Identity8gradients/decoder/dense/LeakyRelu/Maximum_grad/Reshape_1@^gradients/decoder/dense/LeakyRelu/Maximum_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/decoder/dense/LeakyRelu/Maximum_grad/Reshape_1
Y
0gradients/decoder/dense/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
k
2gradients/decoder/dense/LeakyRelu/mul_grad/Shape_1Shapedecoder/dense/BiasAdd*
out_type0*
T0
È
@gradients/decoder/dense/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients/decoder/dense/LeakyRelu/mul_grad/Shape2gradients/decoder/dense/LeakyRelu/mul_grad/Shape_1*
T0

.gradients/decoder/dense/LeakyRelu/mul_grad/mulMulGgradients/decoder/dense/LeakyRelu/Maximum_grad/tuple/control_dependencydecoder/dense/BiasAdd*
T0
Í
.gradients/decoder/dense/LeakyRelu/mul_grad/SumSum.gradients/decoder/dense/LeakyRelu/mul_grad/mul@gradients/decoder/dense/LeakyRelu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
¶
2gradients/decoder/dense/LeakyRelu/mul_grad/ReshapeReshape.gradients/decoder/dense/LeakyRelu/mul_grad/Sum0gradients/decoder/dense/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
¨
0gradients/decoder/dense/LeakyRelu/mul_grad/mul_1Muldecoder/dense/LeakyRelu/alphaGgradients/decoder/dense/LeakyRelu/Maximum_grad/tuple/control_dependency*
T0
Ó
0gradients/decoder/dense/LeakyRelu/mul_grad/Sum_1Sum0gradients/decoder/dense/LeakyRelu/mul_grad/mul_1Bgradients/decoder/dense/LeakyRelu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
¼
4gradients/decoder/dense/LeakyRelu/mul_grad/Reshape_1Reshape0gradients/decoder/dense/LeakyRelu/mul_grad/Sum_12gradients/decoder/dense/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
¯
;gradients/decoder/dense/LeakyRelu/mul_grad/tuple/group_depsNoOp3^gradients/decoder/dense/LeakyRelu/mul_grad/Reshape5^gradients/decoder/dense/LeakyRelu/mul_grad/Reshape_1

Cgradients/decoder/dense/LeakyRelu/mul_grad/tuple/control_dependencyIdentity2gradients/decoder/dense/LeakyRelu/mul_grad/Reshape<^gradients/decoder/dense/LeakyRelu/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/decoder/dense/LeakyRelu/mul_grad/Reshape

Egradients/decoder/dense/LeakyRelu/mul_grad/tuple/control_dependency_1Identity4gradients/decoder/dense/LeakyRelu/mul_grad/Reshape_1<^gradients/decoder/dense/LeakyRelu/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/decoder/dense/LeakyRelu/mul_grad/Reshape_1

gradients/AddN_1AddNIgradients/decoder/dense/LeakyRelu/Maximum_grad/tuple/control_dependency_1Egradients/decoder/dense/LeakyRelu/mul_grad/tuple/control_dependency_1*K
_classA
?=loc:@gradients/decoder/dense/LeakyRelu/Maximum_grad/Reshape_1*
N*
T0
q
0gradients/decoder/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
T0*
data_formatNHWC

5gradients/decoder/dense/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_11^gradients/decoder/dense/BiasAdd_grad/BiasAddGrad
é
=gradients/decoder/dense/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_16^gradients/decoder/dense/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@gradients/decoder/dense/LeakyRelu/Maximum_grad/Reshape_1*
T0

?gradients/decoder/dense/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/decoder/dense/BiasAdd_grad/BiasAddGrad6^gradients/decoder/dense/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/decoder/dense/BiasAdd_grad/BiasAddGrad
½
*gradients/decoder/dense/MatMul_grad/MatMulMatMul=gradients/decoder/dense/BiasAdd_grad/tuple/control_dependencydecoder/dense/kernel/read*
T0*
transpose_a( *
transpose_b(
±
,gradients/decoder/dense/MatMul_grad/MatMul_1MatMulencoder/add=gradients/decoder/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 

4gradients/decoder/dense/MatMul_grad/tuple/group_depsNoOp+^gradients/decoder/dense/MatMul_grad/MatMul-^gradients/decoder/dense/MatMul_grad/MatMul_1
ó
<gradients/decoder/dense/MatMul_grad/tuple/control_dependencyIdentity*gradients/decoder/dense/MatMul_grad/MatMul5^gradients/decoder/dense/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/decoder/dense/MatMul_grad/MatMul
ù
>gradients/decoder/dense/MatMul_grad/tuple/control_dependency_1Identity,gradients/decoder/dense/MatMul_grad/MatMul_15^gradients/decoder/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/decoder/dense/MatMul_grad/MatMul_1
Y
 gradients/encoder/add_grad/ShapeShapeencoder/dense/BiasAdd*
T0*
out_type0
Q
"gradients/encoder/add_grad/Shape_1Shapeencoder/Mul*
T0*
out_type0

0gradients/encoder/add_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/encoder/add_grad/Shape"gradients/encoder/add_grad/Shape_1*
T0
»
gradients/encoder/add_grad/SumSum<gradients/decoder/dense/MatMul_grad/tuple/control_dependency0gradients/encoder/add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0

"gradients/encoder/add_grad/ReshapeReshapegradients/encoder/add_grad/Sum gradients/encoder/add_grad/Shape*
T0*
Tshape0
¿
 gradients/encoder/add_grad/Sum_1Sum<gradients/decoder/dense/MatMul_grad/tuple/control_dependency2gradients/encoder/add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0

$gradients/encoder/add_grad/Reshape_1Reshape gradients/encoder/add_grad/Sum_1"gradients/encoder/add_grad/Shape_1*
T0*
Tshape0

+gradients/encoder/add_grad/tuple/group_depsNoOp#^gradients/encoder/add_grad/Reshape%^gradients/encoder/add_grad/Reshape_1
Ñ
3gradients/encoder/add_grad/tuple/control_dependencyIdentity"gradients/encoder/add_grad/Reshape,^gradients/encoder/add_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/encoder/add_grad/Reshape
×
5gradients/encoder/add_grad/tuple/control_dependency_1Identity$gradients/encoder/add_grad/Reshape_1,^gradients/encoder/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/encoder/add_grad/Reshape_1
¬
gradients/AddN_2AddNgradients/Square_grad/mul_13gradients/encoder/add_grad/tuple/control_dependency*
T0*.
_class$
" loc:@gradients/Square_grad/mul_1*
N
q
0gradients/encoder/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_2*
T0*
data_formatNHWC

5gradients/encoder/dense/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_21^gradients/encoder/dense/BiasAdd_grad/BiasAddGrad
Ì
=gradients/encoder/dense/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_26^gradients/encoder/dense/BiasAdd_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/Square_grad/mul_1

?gradients/encoder/dense/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/encoder/dense/BiasAdd_grad/BiasAddGrad6^gradients/encoder/dense/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/encoder/dense/BiasAdd_grad/BiasAddGrad
Y
 gradients/encoder/Mul_grad/ShapeShapeencoder/random_normal*
T0*
out_type0
Q
"gradients/encoder/Mul_grad/Shape_1Shapeencoder/Exp*
T0*
out_type0

0gradients/encoder/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/encoder/Mul_grad/Shape"gradients/encoder/Mul_grad/Shape_1*
T0
r
gradients/encoder/Mul_grad/mulMul5gradients/encoder/add_grad/tuple/control_dependency_1encoder/Exp*
T0

gradients/encoder/Mul_grad/SumSumgradients/encoder/Mul_grad/mul0gradients/encoder/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0

"gradients/encoder/Mul_grad/ReshapeReshapegradients/encoder/Mul_grad/Sum gradients/encoder/Mul_grad/Shape*
T0*
Tshape0
~
 gradients/encoder/Mul_grad/mul_1Mulencoder/random_normal5gradients/encoder/add_grad/tuple/control_dependency_1*
T0
£
 gradients/encoder/Mul_grad/Sum_1Sum gradients/encoder/Mul_grad/mul_12gradients/encoder/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0

$gradients/encoder/Mul_grad/Reshape_1Reshape gradients/encoder/Mul_grad/Sum_1"gradients/encoder/Mul_grad/Shape_1*
T0*
Tshape0

+gradients/encoder/Mul_grad/tuple/group_depsNoOp#^gradients/encoder/Mul_grad/Reshape%^gradients/encoder/Mul_grad/Reshape_1
Ñ
3gradients/encoder/Mul_grad/tuple/control_dependencyIdentity"gradients/encoder/Mul_grad/Reshape,^gradients/encoder/Mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/encoder/Mul_grad/Reshape
×
5gradients/encoder/Mul_grad/tuple/control_dependency_1Identity$gradients/encoder/Mul_grad/Reshape_1,^gradients/encoder/Mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/encoder/Mul_grad/Reshape_1
½
*gradients/encoder/dense/MatMul_grad/MatMulMatMul=gradients/encoder/dense/BiasAdd_grad/tuple/control_dependencyencoder/dense/kernel/read*
transpose_a( *
transpose_b(*
T0
Å
,gradients/encoder/dense/MatMul_grad/MatMul_1MatMulencoder/Flatten/flatten/Reshape=gradients/encoder/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(

4gradients/encoder/dense/MatMul_grad/tuple/group_depsNoOp+^gradients/encoder/dense/MatMul_grad/MatMul-^gradients/encoder/dense/MatMul_grad/MatMul_1
ó
<gradients/encoder/dense/MatMul_grad/tuple/control_dependencyIdentity*gradients/encoder/dense/MatMul_grad/MatMul5^gradients/encoder/dense/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/encoder/dense/MatMul_grad/MatMul
ù
>gradients/encoder/dense/MatMul_grad/tuple/control_dependency_1Identity,gradients/encoder/dense/MatMul_grad/MatMul_15^gradients/encoder/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/encoder/dense/MatMul_grad/MatMul_1
g
*gradients/encoder/random_normal_grad/ShapeShapeencoder/random_normal/mul*
T0*
out_type0
U
,gradients/encoder/random_normal_grad/Shape_1Const*
valueB *
dtype0
¶
:gradients/encoder/random_normal_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/encoder/random_normal_grad/Shape,gradients/encoder/random_normal_grad/Shape_1*
T0
Æ
(gradients/encoder/random_normal_grad/SumSum3gradients/encoder/Mul_grad/tuple/control_dependency:gradients/encoder/random_normal_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
¤
,gradients/encoder/random_normal_grad/ReshapeReshape(gradients/encoder/random_normal_grad/Sum*gradients/encoder/random_normal_grad/Shape*
Tshape0*
T0
Ê
*gradients/encoder/random_normal_grad/Sum_1Sum3gradients/encoder/Mul_grad/tuple/control_dependency<gradients/encoder/random_normal_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
ª
.gradients/encoder/random_normal_grad/Reshape_1Reshape*gradients/encoder/random_normal_grad/Sum_1,gradients/encoder/random_normal_grad/Shape_1*
T0*
Tshape0

5gradients/encoder/random_normal_grad/tuple/group_depsNoOp-^gradients/encoder/random_normal_grad/Reshape/^gradients/encoder/random_normal_grad/Reshape_1
ù
=gradients/encoder/random_normal_grad/tuple/control_dependencyIdentity,gradients/encoder/random_normal_grad/Reshape6^gradients/encoder/random_normal_grad/tuple/group_deps*?
_class5
31loc:@gradients/encoder/random_normal_grad/Reshape*
T0
ÿ
?gradients/encoder/random_normal_grad/tuple/control_dependency_1Identity.gradients/encoder/random_normal_grad/Reshape_16^gradients/encoder/random_normal_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/encoder/random_normal_grad/Reshape_1
r
gradients/encoder/Exp_grad/mulMul5gradients/encoder/Mul_grad/tuple/control_dependency_1encoder/Exp*
T0
|
.gradients/encoder/random_normal/mul_grad/ShapeShape*encoder/random_normal/RandomStandardNormal*
T0*
out_type0
Y
0gradients/encoder/random_normal/mul_grad/Shape_1Const*
valueB *
dtype0
Â
>gradients/encoder/random_normal/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/encoder/random_normal/mul_grad/Shape0gradients/encoder/random_normal/mul_grad/Shape_1*
T0

,gradients/encoder/random_normal/mul_grad/mulMul=gradients/encoder/random_normal_grad/tuple/control_dependencyencoder/random_normal/stddev*
T0
Ç
,gradients/encoder/random_normal/mul_grad/SumSum,gradients/encoder/random_normal/mul_grad/mul>gradients/encoder/random_normal/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
°
0gradients/encoder/random_normal/mul_grad/ReshapeReshape,gradients/encoder/random_normal/mul_grad/Sum.gradients/encoder/random_normal/mul_grad/Shape*
T0*
Tshape0
©
.gradients/encoder/random_normal/mul_grad/mul_1Mul*encoder/random_normal/RandomStandardNormal=gradients/encoder/random_normal_grad/tuple/control_dependency*
T0
Í
.gradients/encoder/random_normal/mul_grad/Sum_1Sum.gradients/encoder/random_normal/mul_grad/mul_1@gradients/encoder/random_normal/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
¶
2gradients/encoder/random_normal/mul_grad/Reshape_1Reshape.gradients/encoder/random_normal/mul_grad/Sum_10gradients/encoder/random_normal/mul_grad/Shape_1*
T0*
Tshape0
©
9gradients/encoder/random_normal/mul_grad/tuple/group_depsNoOp1^gradients/encoder/random_normal/mul_grad/Reshape3^gradients/encoder/random_normal/mul_grad/Reshape_1

Agradients/encoder/random_normal/mul_grad/tuple/control_dependencyIdentity0gradients/encoder/random_normal/mul_grad/Reshape:^gradients/encoder/random_normal/mul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/encoder/random_normal/mul_grad/Reshape

Cgradients/encoder/random_normal/mul_grad/tuple/control_dependency_1Identity2gradients/encoder/random_normal/mul_grad/Reshape_1:^gradients/encoder/random_normal/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/encoder/random_normal/mul_grad/Reshape_1
Ý
gradients/AddN_3AddN/gradients/mul_1_grad/tuple/control_dependency_1-gradients/mul_grad/tuple/control_dependency_1gradients/encoder/Exp_grad/mul*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1*
N*
T0
I
 gradients/encoder/mul_grad/ShapeConst*
valueB *
dtype0
]
"gradients/encoder/mul_grad/Shape_1Shapeencoder/dense_2/BiasAdd*
T0*
out_type0

0gradients/encoder/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/encoder/mul_grad/Shape"gradients/encoder/mul_grad/Shape_1*
T0
Y
gradients/encoder/mul_grad/mulMulgradients/AddN_3encoder/dense_2/BiasAdd*
T0

gradients/encoder/mul_grad/SumSumgradients/encoder/mul_grad/mul0gradients/encoder/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0

"gradients/encoder/mul_grad/ReshapeReshapegradients/encoder/mul_grad/Sum gradients/encoder/mul_grad/Shape*
T0*
Tshape0
Q
 gradients/encoder/mul_grad/mul_1Mulencoder/mul/xgradients/AddN_3*
T0
£
 gradients/encoder/mul_grad/Sum_1Sum gradients/encoder/mul_grad/mul_12gradients/encoder/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0

$gradients/encoder/mul_grad/Reshape_1Reshape gradients/encoder/mul_grad/Sum_1"gradients/encoder/mul_grad/Shape_1*
Tshape0*
T0

+gradients/encoder/mul_grad/tuple/group_depsNoOp#^gradients/encoder/mul_grad/Reshape%^gradients/encoder/mul_grad/Reshape_1
Ñ
3gradients/encoder/mul_grad/tuple/control_dependencyIdentity"gradients/encoder/mul_grad/Reshape,^gradients/encoder/mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/encoder/mul_grad/Reshape
×
5gradients/encoder/mul_grad/tuple/control_dependency_1Identity$gradients/encoder/mul_grad/Reshape_1,^gradients/encoder/mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/encoder/mul_grad/Reshape_1

2gradients/encoder/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/encoder/mul_grad/tuple/control_dependency_1*
T0*
data_formatNHWC
¬
7gradients/encoder/dense_2/BiasAdd_grad/tuple/group_depsNoOp6^gradients/encoder/mul_grad/tuple/control_dependency_13^gradients/encoder/dense_2/BiasAdd_grad/BiasAddGrad
þ
?gradients/encoder/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/encoder/mul_grad/tuple/control_dependency_18^gradients/encoder/dense_2/BiasAdd_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/encoder/mul_grad/Reshape_1

Agradients/encoder/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/encoder/dense_2/BiasAdd_grad/BiasAddGrad8^gradients/encoder/dense_2/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/encoder/dense_2/BiasAdd_grad/BiasAddGrad
Ã
,gradients/encoder/dense_2/MatMul_grad/MatMulMatMul?gradients/encoder/dense_2/BiasAdd_grad/tuple/control_dependencyencoder/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b(
É
.gradients/encoder/dense_2/MatMul_grad/MatMul_1MatMulencoder/Flatten/flatten/Reshape?gradients/encoder/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0

6gradients/encoder/dense_2/MatMul_grad/tuple/group_depsNoOp-^gradients/encoder/dense_2/MatMul_grad/MatMul/^gradients/encoder/dense_2/MatMul_grad/MatMul_1
û
>gradients/encoder/dense_2/MatMul_grad/tuple/control_dependencyIdentity,gradients/encoder/dense_2/MatMul_grad/MatMul7^gradients/encoder/dense_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/encoder/dense_2/MatMul_grad/MatMul

@gradients/encoder/dense_2/MatMul_grad/tuple/control_dependency_1Identity.gradients/encoder/dense_2/MatMul_grad/MatMul_17^gradients/encoder/dense_2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/encoder/dense_2/MatMul_grad/MatMul_1
ç
gradients/AddN_4AddN<gradients/encoder/dense/MatMul_grad/tuple/control_dependency>gradients/encoder/dense_2/MatMul_grad/tuple/control_dependency*
T0*=
_class3
1/loc:@gradients/encoder/dense/MatMul_grad/MatMul*
N
m
4gradients/encoder/Flatten/flatten/Reshape_grad/ShapeShapeencoder/dropout_2/mul*
out_type0*
T0
 
6gradients/encoder/Flatten/flatten/Reshape_grad/ReshapeReshapegradients/AddN_44gradients/encoder/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0
c
*gradients/encoder/dropout_2/mul_grad/ShapeShapeencoder/dropout_2/div*
T0*
out_type0
g
,gradients/encoder/dropout_2/mul_grad/Shape_1Shapeencoder/dropout_2/Floor*
T0*
out_type0
¶
:gradients/encoder/dropout_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/encoder/dropout_2/mul_grad/Shape,gradients/encoder/dropout_2/mul_grad/Shape_1*
T0

(gradients/encoder/dropout_2/mul_grad/mulMul6gradients/encoder/Flatten/flatten/Reshape_grad/Reshapeencoder/dropout_2/Floor*
T0
»
(gradients/encoder/dropout_2/mul_grad/SumSum(gradients/encoder/dropout_2/mul_grad/mul:gradients/encoder/dropout_2/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
¤
,gradients/encoder/dropout_2/mul_grad/ReshapeReshape(gradients/encoder/dropout_2/mul_grad/Sum*gradients/encoder/dropout_2/mul_grad/Shape*
T0*
Tshape0

*gradients/encoder/dropout_2/mul_grad/mul_1Mulencoder/dropout_2/div6gradients/encoder/Flatten/flatten/Reshape_grad/Reshape*
T0
Á
*gradients/encoder/dropout_2/mul_grad/Sum_1Sum*gradients/encoder/dropout_2/mul_grad/mul_1<gradients/encoder/dropout_2/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
ª
.gradients/encoder/dropout_2/mul_grad/Reshape_1Reshape*gradients/encoder/dropout_2/mul_grad/Sum_1,gradients/encoder/dropout_2/mul_grad/Shape_1*
T0*
Tshape0

5gradients/encoder/dropout_2/mul_grad/tuple/group_depsNoOp-^gradients/encoder/dropout_2/mul_grad/Reshape/^gradients/encoder/dropout_2/mul_grad/Reshape_1
ù
=gradients/encoder/dropout_2/mul_grad/tuple/control_dependencyIdentity,gradients/encoder/dropout_2/mul_grad/Reshape6^gradients/encoder/dropout_2/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/encoder/dropout_2/mul_grad/Reshape
ÿ
?gradients/encoder/dropout_2/mul_grad/tuple/control_dependency_1Identity.gradients/encoder/dropout_2/mul_grad/Reshape_16^gradients/encoder/dropout_2/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/encoder/dropout_2/mul_grad/Reshape_1
p
*gradients/encoder/dropout_2/div_grad/ShapeShape"encoder/conv2d_3/LeakyRelu/Maximum*
T0*
out_type0
U
,gradients/encoder/dropout_2/div_grad/Shape_1Const*
valueB *
dtype0
¶
:gradients/encoder/dropout_2/div_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/encoder/dropout_2/div_grad/Shape,gradients/encoder/dropout_2/div_grad/Shape_1*
T0

,gradients/encoder/dropout_2/div_grad/RealDivRealDiv=gradients/encoder/dropout_2/mul_grad/tuple/control_dependency	keep_prob*
T0
¿
(gradients/encoder/dropout_2/div_grad/SumSum,gradients/encoder/dropout_2/div_grad/RealDiv:gradients/encoder/dropout_2/div_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
¤
,gradients/encoder/dropout_2/div_grad/ReshapeReshape(gradients/encoder/dropout_2/div_grad/Sum*gradients/encoder/dropout_2/div_grad/Shape*
T0*
Tshape0
\
(gradients/encoder/dropout_2/div_grad/NegNeg"encoder/conv2d_3/LeakyRelu/Maximum*
T0
w
.gradients/encoder/dropout_2/div_grad/RealDiv_1RealDiv(gradients/encoder/dropout_2/div_grad/Neg	keep_prob*
T0
}
.gradients/encoder/dropout_2/div_grad/RealDiv_2RealDiv.gradients/encoder/dropout_2/div_grad/RealDiv_1	keep_prob*
T0
§
(gradients/encoder/dropout_2/div_grad/mulMul=gradients/encoder/dropout_2/mul_grad/tuple/control_dependency.gradients/encoder/dropout_2/div_grad/RealDiv_2*
T0
¿
*gradients/encoder/dropout_2/div_grad/Sum_1Sum(gradients/encoder/dropout_2/div_grad/mul<gradients/encoder/dropout_2/div_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
ª
.gradients/encoder/dropout_2/div_grad/Reshape_1Reshape*gradients/encoder/dropout_2/div_grad/Sum_1,gradients/encoder/dropout_2/div_grad/Shape_1*
T0*
Tshape0

5gradients/encoder/dropout_2/div_grad/tuple/group_depsNoOp-^gradients/encoder/dropout_2/div_grad/Reshape/^gradients/encoder/dropout_2/div_grad/Reshape_1
ù
=gradients/encoder/dropout_2/div_grad/tuple/control_dependencyIdentity,gradients/encoder/dropout_2/div_grad/Reshape6^gradients/encoder/dropout_2/div_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/encoder/dropout_2/div_grad/Reshape
ÿ
?gradients/encoder/dropout_2/div_grad/tuple/control_dependency_1Identity.gradients/encoder/dropout_2/div_grad/Reshape_16^gradients/encoder/dropout_2/div_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/encoder/dropout_2/div_grad/Reshape_1
y
7gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/ShapeShapeencoder/conv2d_3/LeakyRelu/mul*
T0*
out_type0
u
9gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Shape_1Shapeencoder/conv2d_3/BiasAdd*
T0*
out_type0

9gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Shape_2Shape=gradients/encoder/dropout_2/div_grad/tuple/control_dependency*
out_type0*
T0
j
=gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
Â
7gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/zerosFill9gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Shape_2=gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/zeros/Const*
T0

>gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/GreaterEqualGreaterEqualencoder/conv2d_3/LeakyRelu/mulencoder/conv2d_3/BiasAdd*
T0
Ý
Ggradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Shape9gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Shape_1*
T0

8gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/SelectSelect>gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/GreaterEqual=gradients/encoder/dropout_2/div_grad/tuple/control_dependency7gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/zeros*
T0

:gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Select_1Select>gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/GreaterEqual7gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/zeros=gradients/encoder/dropout_2/div_grad/tuple/control_dependency*
T0
å
5gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/SumSum8gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/SelectGgradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
Ë
9gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/ReshapeReshape5gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Sum7gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Shape*
T0*
Tshape0
ë
7gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Sum_1Sum:gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Select_1Igradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
Ñ
;gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Reshape_1Reshape7gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Sum_19gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Shape_1*
T0*
Tshape0
Ä
Bgradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/tuple/group_depsNoOp:^gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Reshape<^gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Reshape_1
­
Jgradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/tuple/control_dependencyIdentity9gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/ReshapeC^gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Reshape
³
Lgradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/tuple/control_dependency_1Identity;gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Reshape_1C^gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Reshape_1
\
3gradients/encoder/conv2d_3/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
q
5gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Shape_1Shapeencoder/conv2d_3/BiasAdd*
T0*
out_type0
Ñ
Cgradients/encoder/conv2d_3/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Shape5gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Shape_1*
T0
§
1gradients/encoder/conv2d_3/LeakyRelu/mul_grad/mulMulJgradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/tuple/control_dependencyencoder/conv2d_3/BiasAdd*
T0
Ö
1gradients/encoder/conv2d_3/LeakyRelu/mul_grad/SumSum1gradients/encoder/conv2d_3/LeakyRelu/mul_grad/mulCgradients/encoder/conv2d_3/LeakyRelu/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
¿
5gradients/encoder/conv2d_3/LeakyRelu/mul_grad/ReshapeReshape1gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Sum3gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
±
3gradients/encoder/conv2d_3/LeakyRelu/mul_grad/mul_1Mul encoder/conv2d_3/LeakyRelu/alphaJgradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/tuple/control_dependency*
T0
Ü
3gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Sum_1Sum3gradients/encoder/conv2d_3/LeakyRelu/mul_grad/mul_1Egradients/encoder/conv2d_3/LeakyRelu/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
Å
7gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Reshape_1Reshape3gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Sum_15gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
¸
>gradients/encoder/conv2d_3/LeakyRelu/mul_grad/tuple/group_depsNoOp6^gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Reshape8^gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Reshape_1

Fgradients/encoder/conv2d_3/LeakyRelu/mul_grad/tuple/control_dependencyIdentity5gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Reshape?^gradients/encoder/conv2d_3/LeakyRelu/mul_grad/tuple/group_deps*H
_class>
<:loc:@gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Reshape*
T0
£
Hgradients/encoder/conv2d_3/LeakyRelu/mul_grad/tuple/control_dependency_1Identity7gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Reshape_1?^gradients/encoder/conv2d_3/LeakyRelu/mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/encoder/conv2d_3/LeakyRelu/mul_grad/Reshape_1

gradients/AddN_5AddNLgradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/tuple/control_dependency_1Hgradients/encoder/conv2d_3/LeakyRelu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Reshape_1*
N
t
3gradients/encoder/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
data_formatNHWC*
T0

8gradients/encoder/conv2d_3/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_54^gradients/encoder/conv2d_3/BiasAdd_grad/BiasAddGrad
ò
@gradients/encoder/conv2d_3/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_59^gradients/encoder/conv2d_3/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/encoder/conv2d_3/LeakyRelu/Maximum_grad/Reshape_1

Bgradients/encoder/conv2d_3/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/encoder/conv2d_3/BiasAdd_grad/BiasAddGrad9^gradients/encoder/conv2d_3/BiasAdd_grad/tuple/group_deps*F
_class<
:8loc:@gradients/encoder/conv2d_3/BiasAdd_grad/BiasAddGrad*
T0

-gradients/encoder/conv2d_3/Conv2D_grad/ShapeNShapeNencoder/dropout_1/mulencoder/conv2d_2/kernel/read*
N*
T0*
out_type0
À
:gradients/encoder/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput-gradients/encoder/conv2d_3/Conv2D_grad/ShapeNencoder/conv2d_2/kernel/read@gradients/encoder/conv2d_3/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
T0
½
;gradients/encoder/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/dropout_1/mul/gradients/encoder/conv2d_3/Conv2D_grad/ShapeN:1@gradients/encoder/conv2d_3/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
º
7gradients/encoder/conv2d_3/Conv2D_grad/tuple/group_depsNoOp;^gradients/encoder/conv2d_3/Conv2D_grad/Conv2DBackpropInput<^gradients/encoder/conv2d_3/Conv2D_grad/Conv2DBackpropFilter

?gradients/encoder/conv2d_3/Conv2D_grad/tuple/control_dependencyIdentity:gradients/encoder/conv2d_3/Conv2D_grad/Conv2DBackpropInput8^gradients/encoder/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/encoder/conv2d_3/Conv2D_grad/Conv2DBackpropInput

Agradients/encoder/conv2d_3/Conv2D_grad/tuple/control_dependency_1Identity;gradients/encoder/conv2d_3/Conv2D_grad/Conv2DBackpropFilter8^gradients/encoder/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/encoder/conv2d_3/Conv2D_grad/Conv2DBackpropFilter
c
*gradients/encoder/dropout_1/mul_grad/ShapeShapeencoder/dropout_1/div*
out_type0*
T0
g
,gradients/encoder/dropout_1/mul_grad/Shape_1Shapeencoder/dropout_1/Floor*
out_type0*
T0
¶
:gradients/encoder/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/encoder/dropout_1/mul_grad/Shape,gradients/encoder/dropout_1/mul_grad/Shape_1*
T0

(gradients/encoder/dropout_1/mul_grad/mulMul?gradients/encoder/conv2d_3/Conv2D_grad/tuple/control_dependencyencoder/dropout_1/Floor*
T0
»
(gradients/encoder/dropout_1/mul_grad/SumSum(gradients/encoder/dropout_1/mul_grad/mul:gradients/encoder/dropout_1/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
¤
,gradients/encoder/dropout_1/mul_grad/ReshapeReshape(gradients/encoder/dropout_1/mul_grad/Sum*gradients/encoder/dropout_1/mul_grad/Shape*
T0*
Tshape0

*gradients/encoder/dropout_1/mul_grad/mul_1Mulencoder/dropout_1/div?gradients/encoder/conv2d_3/Conv2D_grad/tuple/control_dependency*
T0
Á
*gradients/encoder/dropout_1/mul_grad/Sum_1Sum*gradients/encoder/dropout_1/mul_grad/mul_1<gradients/encoder/dropout_1/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
ª
.gradients/encoder/dropout_1/mul_grad/Reshape_1Reshape*gradients/encoder/dropout_1/mul_grad/Sum_1,gradients/encoder/dropout_1/mul_grad/Shape_1*
T0*
Tshape0

5gradients/encoder/dropout_1/mul_grad/tuple/group_depsNoOp-^gradients/encoder/dropout_1/mul_grad/Reshape/^gradients/encoder/dropout_1/mul_grad/Reshape_1
ù
=gradients/encoder/dropout_1/mul_grad/tuple/control_dependencyIdentity,gradients/encoder/dropout_1/mul_grad/Reshape6^gradients/encoder/dropout_1/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/encoder/dropout_1/mul_grad/Reshape
ÿ
?gradients/encoder/dropout_1/mul_grad/tuple/control_dependency_1Identity.gradients/encoder/dropout_1/mul_grad/Reshape_16^gradients/encoder/dropout_1/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/encoder/dropout_1/mul_grad/Reshape_1
p
*gradients/encoder/dropout_1/div_grad/ShapeShape"encoder/conv2d_2/LeakyRelu/Maximum*
out_type0*
T0
U
,gradients/encoder/dropout_1/div_grad/Shape_1Const*
valueB *
dtype0
¶
:gradients/encoder/dropout_1/div_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/encoder/dropout_1/div_grad/Shape,gradients/encoder/dropout_1/div_grad/Shape_1*
T0

,gradients/encoder/dropout_1/div_grad/RealDivRealDiv=gradients/encoder/dropout_1/mul_grad/tuple/control_dependency	keep_prob*
T0
¿
(gradients/encoder/dropout_1/div_grad/SumSum,gradients/encoder/dropout_1/div_grad/RealDiv:gradients/encoder/dropout_1/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
¤
,gradients/encoder/dropout_1/div_grad/ReshapeReshape(gradients/encoder/dropout_1/div_grad/Sum*gradients/encoder/dropout_1/div_grad/Shape*
T0*
Tshape0
\
(gradients/encoder/dropout_1/div_grad/NegNeg"encoder/conv2d_2/LeakyRelu/Maximum*
T0
w
.gradients/encoder/dropout_1/div_grad/RealDiv_1RealDiv(gradients/encoder/dropout_1/div_grad/Neg	keep_prob*
T0
}
.gradients/encoder/dropout_1/div_grad/RealDiv_2RealDiv.gradients/encoder/dropout_1/div_grad/RealDiv_1	keep_prob*
T0
§
(gradients/encoder/dropout_1/div_grad/mulMul=gradients/encoder/dropout_1/mul_grad/tuple/control_dependency.gradients/encoder/dropout_1/div_grad/RealDiv_2*
T0
¿
*gradients/encoder/dropout_1/div_grad/Sum_1Sum(gradients/encoder/dropout_1/div_grad/mul<gradients/encoder/dropout_1/div_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
ª
.gradients/encoder/dropout_1/div_grad/Reshape_1Reshape*gradients/encoder/dropout_1/div_grad/Sum_1,gradients/encoder/dropout_1/div_grad/Shape_1*
T0*
Tshape0

5gradients/encoder/dropout_1/div_grad/tuple/group_depsNoOp-^gradients/encoder/dropout_1/div_grad/Reshape/^gradients/encoder/dropout_1/div_grad/Reshape_1
ù
=gradients/encoder/dropout_1/div_grad/tuple/control_dependencyIdentity,gradients/encoder/dropout_1/div_grad/Reshape6^gradients/encoder/dropout_1/div_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/encoder/dropout_1/div_grad/Reshape
ÿ
?gradients/encoder/dropout_1/div_grad/tuple/control_dependency_1Identity.gradients/encoder/dropout_1/div_grad/Reshape_16^gradients/encoder/dropout_1/div_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/encoder/dropout_1/div_grad/Reshape_1
y
7gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/ShapeShapeencoder/conv2d_2/LeakyRelu/mul*
T0*
out_type0
u
9gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Shape_1Shapeencoder/conv2d_2/BiasAdd*
T0*
out_type0

9gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Shape_2Shape=gradients/encoder/dropout_1/div_grad/tuple/control_dependency*
T0*
out_type0
j
=gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
Â
7gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/zerosFill9gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Shape_2=gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/zeros/Const*
T0

>gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/GreaterEqualGreaterEqualencoder/conv2d_2/LeakyRelu/mulencoder/conv2d_2/BiasAdd*
T0
Ý
Ggradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Shape9gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Shape_1*
T0

8gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/SelectSelect>gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/GreaterEqual=gradients/encoder/dropout_1/div_grad/tuple/control_dependency7gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/zeros*
T0

:gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Select_1Select>gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/GreaterEqual7gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/zeros=gradients/encoder/dropout_1/div_grad/tuple/control_dependency*
T0
å
5gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/SumSum8gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/SelectGgradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
Ë
9gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/ReshapeReshape5gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Sum7gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Shape*
T0*
Tshape0
ë
7gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Sum_1Sum:gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Select_1Igradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
Ñ
;gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Reshape_1Reshape7gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Sum_19gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Shape_1*
T0*
Tshape0
Ä
Bgradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/tuple/group_depsNoOp:^gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Reshape<^gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Reshape_1
­
Jgradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/tuple/control_dependencyIdentity9gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/ReshapeC^gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Reshape
³
Lgradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/tuple/control_dependency_1Identity;gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Reshape_1C^gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Reshape_1
\
3gradients/encoder/conv2d_2/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
q
5gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Shape_1Shapeencoder/conv2d_2/BiasAdd*
T0*
out_type0
Ñ
Cgradients/encoder/conv2d_2/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Shape5gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Shape_1*
T0
§
1gradients/encoder/conv2d_2/LeakyRelu/mul_grad/mulMulJgradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/tuple/control_dependencyencoder/conv2d_2/BiasAdd*
T0
Ö
1gradients/encoder/conv2d_2/LeakyRelu/mul_grad/SumSum1gradients/encoder/conv2d_2/LeakyRelu/mul_grad/mulCgradients/encoder/conv2d_2/LeakyRelu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
¿
5gradients/encoder/conv2d_2/LeakyRelu/mul_grad/ReshapeReshape1gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Sum3gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Shape*
Tshape0*
T0
±
3gradients/encoder/conv2d_2/LeakyRelu/mul_grad/mul_1Mul encoder/conv2d_2/LeakyRelu/alphaJgradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/tuple/control_dependency*
T0
Ü
3gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Sum_1Sum3gradients/encoder/conv2d_2/LeakyRelu/mul_grad/mul_1Egradients/encoder/conv2d_2/LeakyRelu/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
Å
7gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Reshape_1Reshape3gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Sum_15gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Shape_1*
Tshape0*
T0
¸
>gradients/encoder/conv2d_2/LeakyRelu/mul_grad/tuple/group_depsNoOp6^gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Reshape8^gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Reshape_1

Fgradients/encoder/conv2d_2/LeakyRelu/mul_grad/tuple/control_dependencyIdentity5gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Reshape?^gradients/encoder/conv2d_2/LeakyRelu/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Reshape
£
Hgradients/encoder/conv2d_2/LeakyRelu/mul_grad/tuple/control_dependency_1Identity7gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Reshape_1?^gradients/encoder/conv2d_2/LeakyRelu/mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/encoder/conv2d_2/LeakyRelu/mul_grad/Reshape_1

gradients/AddN_6AddNLgradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/tuple/control_dependency_1Hgradients/encoder/conv2d_2/LeakyRelu/mul_grad/tuple/control_dependency_1*N
_classD
B@loc:@gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Reshape_1*
N*
T0
t
3gradients/encoder/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
T0*
data_formatNHWC

8gradients/encoder/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_64^gradients/encoder/conv2d_2/BiasAdd_grad/BiasAddGrad
ò
@gradients/encoder/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_69^gradients/encoder/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/encoder/conv2d_2/LeakyRelu/Maximum_grad/Reshape_1

Bgradients/encoder/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/encoder/conv2d_2/BiasAdd_grad/BiasAddGrad9^gradients/encoder/conv2d_2/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/encoder/conv2d_2/BiasAdd_grad/BiasAddGrad

-gradients/encoder/conv2d_2/Conv2D_grad/ShapeNShapeNencoder/dropout/mulencoder/conv2d_1/kernel/read*
N*
T0*
out_type0
À
:gradients/encoder/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput-gradients/encoder/conv2d_2/Conv2D_grad/ShapeNencoder/conv2d_1/kernel/read@gradients/encoder/conv2d_2/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
»
;gradients/encoder/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/dropout/mul/gradients/encoder/conv2d_2/Conv2D_grad/ShapeN:1@gradients/encoder/conv2d_2/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
º
7gradients/encoder/conv2d_2/Conv2D_grad/tuple/group_depsNoOp;^gradients/encoder/conv2d_2/Conv2D_grad/Conv2DBackpropInput<^gradients/encoder/conv2d_2/Conv2D_grad/Conv2DBackpropFilter

?gradients/encoder/conv2d_2/Conv2D_grad/tuple/control_dependencyIdentity:gradients/encoder/conv2d_2/Conv2D_grad/Conv2DBackpropInput8^gradients/encoder/conv2d_2/Conv2D_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/encoder/conv2d_2/Conv2D_grad/Conv2DBackpropInput

Agradients/encoder/conv2d_2/Conv2D_grad/tuple/control_dependency_1Identity;gradients/encoder/conv2d_2/Conv2D_grad/Conv2DBackpropFilter8^gradients/encoder/conv2d_2/Conv2D_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/encoder/conv2d_2/Conv2D_grad/Conv2DBackpropFilter
_
(gradients/encoder/dropout/mul_grad/ShapeShapeencoder/dropout/div*
T0*
out_type0
c
*gradients/encoder/dropout/mul_grad/Shape_1Shapeencoder/dropout/Floor*
T0*
out_type0
°
8gradients/encoder/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/encoder/dropout/mul_grad/Shape*gradients/encoder/dropout/mul_grad/Shape_1*
T0

&gradients/encoder/dropout/mul_grad/mulMul?gradients/encoder/conv2d_2/Conv2D_grad/tuple/control_dependencyencoder/dropout/Floor*
T0
µ
&gradients/encoder/dropout/mul_grad/SumSum&gradients/encoder/dropout/mul_grad/mul8gradients/encoder/dropout/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0

*gradients/encoder/dropout/mul_grad/ReshapeReshape&gradients/encoder/dropout/mul_grad/Sum(gradients/encoder/dropout/mul_grad/Shape*
T0*
Tshape0

(gradients/encoder/dropout/mul_grad/mul_1Mulencoder/dropout/div?gradients/encoder/conv2d_2/Conv2D_grad/tuple/control_dependency*
T0
»
(gradients/encoder/dropout/mul_grad/Sum_1Sum(gradients/encoder/dropout/mul_grad/mul_1:gradients/encoder/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
¤
,gradients/encoder/dropout/mul_grad/Reshape_1Reshape(gradients/encoder/dropout/mul_grad/Sum_1*gradients/encoder/dropout/mul_grad/Shape_1*
T0*
Tshape0

3gradients/encoder/dropout/mul_grad/tuple/group_depsNoOp+^gradients/encoder/dropout/mul_grad/Reshape-^gradients/encoder/dropout/mul_grad/Reshape_1
ñ
;gradients/encoder/dropout/mul_grad/tuple/control_dependencyIdentity*gradients/encoder/dropout/mul_grad/Reshape4^gradients/encoder/dropout/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/encoder/dropout/mul_grad/Reshape
÷
=gradients/encoder/dropout/mul_grad/tuple/control_dependency_1Identity,gradients/encoder/dropout/mul_grad/Reshape_14^gradients/encoder/dropout/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/encoder/dropout/mul_grad/Reshape_1
l
(gradients/encoder/dropout/div_grad/ShapeShape encoder/conv2d/LeakyRelu/Maximum*
T0*
out_type0
S
*gradients/encoder/dropout/div_grad/Shape_1Const*
valueB *
dtype0
°
8gradients/encoder/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/encoder/dropout/div_grad/Shape*gradients/encoder/dropout/div_grad/Shape_1*
T0

*gradients/encoder/dropout/div_grad/RealDivRealDiv;gradients/encoder/dropout/mul_grad/tuple/control_dependency	keep_prob*
T0
¹
&gradients/encoder/dropout/div_grad/SumSum*gradients/encoder/dropout/div_grad/RealDiv8gradients/encoder/dropout/div_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0

*gradients/encoder/dropout/div_grad/ReshapeReshape&gradients/encoder/dropout/div_grad/Sum(gradients/encoder/dropout/div_grad/Shape*
T0*
Tshape0
X
&gradients/encoder/dropout/div_grad/NegNeg encoder/conv2d/LeakyRelu/Maximum*
T0
s
,gradients/encoder/dropout/div_grad/RealDiv_1RealDiv&gradients/encoder/dropout/div_grad/Neg	keep_prob*
T0
y
,gradients/encoder/dropout/div_grad/RealDiv_2RealDiv,gradients/encoder/dropout/div_grad/RealDiv_1	keep_prob*
T0
¡
&gradients/encoder/dropout/div_grad/mulMul;gradients/encoder/dropout/mul_grad/tuple/control_dependency,gradients/encoder/dropout/div_grad/RealDiv_2*
T0
¹
(gradients/encoder/dropout/div_grad/Sum_1Sum&gradients/encoder/dropout/div_grad/mul:gradients/encoder/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
¤
,gradients/encoder/dropout/div_grad/Reshape_1Reshape(gradients/encoder/dropout/div_grad/Sum_1*gradients/encoder/dropout/div_grad/Shape_1*
T0*
Tshape0

3gradients/encoder/dropout/div_grad/tuple/group_depsNoOp+^gradients/encoder/dropout/div_grad/Reshape-^gradients/encoder/dropout/div_grad/Reshape_1
ñ
;gradients/encoder/dropout/div_grad/tuple/control_dependencyIdentity*gradients/encoder/dropout/div_grad/Reshape4^gradients/encoder/dropout/div_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/encoder/dropout/div_grad/Reshape
÷
=gradients/encoder/dropout/div_grad/tuple/control_dependency_1Identity,gradients/encoder/dropout/div_grad/Reshape_14^gradients/encoder/dropout/div_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/encoder/dropout/div_grad/Reshape_1
u
5gradients/encoder/conv2d/LeakyRelu/Maximum_grad/ShapeShapeencoder/conv2d/LeakyRelu/mul*
T0*
out_type0
q
7gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Shape_1Shapeencoder/conv2d/BiasAdd*
T0*
out_type0

7gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Shape_2Shape;gradients/encoder/dropout/div_grad/tuple/control_dependency*
out_type0*
T0
h
;gradients/encoder/conv2d/LeakyRelu/Maximum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
¼
5gradients/encoder/conv2d/LeakyRelu/Maximum_grad/zerosFill7gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Shape_2;gradients/encoder/conv2d/LeakyRelu/Maximum_grad/zeros/Const*
T0

<gradients/encoder/conv2d/LeakyRelu/Maximum_grad/GreaterEqualGreaterEqualencoder/conv2d/LeakyRelu/mulencoder/conv2d/BiasAdd*
T0
×
Egradients/encoder/conv2d/LeakyRelu/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Shape7gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Shape_1*
T0
û
6gradients/encoder/conv2d/LeakyRelu/Maximum_grad/SelectSelect<gradients/encoder/conv2d/LeakyRelu/Maximum_grad/GreaterEqual;gradients/encoder/dropout/div_grad/tuple/control_dependency5gradients/encoder/conv2d/LeakyRelu/Maximum_grad/zeros*
T0
ý
8gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Select_1Select<gradients/encoder/conv2d/LeakyRelu/Maximum_grad/GreaterEqual5gradients/encoder/conv2d/LeakyRelu/Maximum_grad/zeros;gradients/encoder/dropout/div_grad/tuple/control_dependency*
T0
ß
3gradients/encoder/conv2d/LeakyRelu/Maximum_grad/SumSum6gradients/encoder/conv2d/LeakyRelu/Maximum_grad/SelectEgradients/encoder/conv2d/LeakyRelu/Maximum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
Å
7gradients/encoder/conv2d/LeakyRelu/Maximum_grad/ReshapeReshape3gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Sum5gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Shape*
Tshape0*
T0
å
5gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Sum_1Sum8gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Select_1Ggradients/encoder/conv2d/LeakyRelu/Maximum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
Ë
9gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Reshape_1Reshape5gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Sum_17gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Shape_1*
T0*
Tshape0
¾
@gradients/encoder/conv2d/LeakyRelu/Maximum_grad/tuple/group_depsNoOp8^gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Reshape:^gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Reshape_1
¥
Hgradients/encoder/conv2d/LeakyRelu/Maximum_grad/tuple/control_dependencyIdentity7gradients/encoder/conv2d/LeakyRelu/Maximum_grad/ReshapeA^gradients/encoder/conv2d/LeakyRelu/Maximum_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Reshape
«
Jgradients/encoder/conv2d/LeakyRelu/Maximum_grad/tuple/control_dependency_1Identity9gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Reshape_1A^gradients/encoder/conv2d/LeakyRelu/Maximum_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Reshape_1
Z
1gradients/encoder/conv2d/LeakyRelu/mul_grad/ShapeConst*
valueB *
dtype0
m
3gradients/encoder/conv2d/LeakyRelu/mul_grad/Shape_1Shapeencoder/conv2d/BiasAdd*
T0*
out_type0
Ë
Agradients/encoder/conv2d/LeakyRelu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/encoder/conv2d/LeakyRelu/mul_grad/Shape3gradients/encoder/conv2d/LeakyRelu/mul_grad/Shape_1*
T0
¡
/gradients/encoder/conv2d/LeakyRelu/mul_grad/mulMulHgradients/encoder/conv2d/LeakyRelu/Maximum_grad/tuple/control_dependencyencoder/conv2d/BiasAdd*
T0
Ð
/gradients/encoder/conv2d/LeakyRelu/mul_grad/SumSum/gradients/encoder/conv2d/LeakyRelu/mul_grad/mulAgradients/encoder/conv2d/LeakyRelu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
¹
3gradients/encoder/conv2d/LeakyRelu/mul_grad/ReshapeReshape/gradients/encoder/conv2d/LeakyRelu/mul_grad/Sum1gradients/encoder/conv2d/LeakyRelu/mul_grad/Shape*
T0*
Tshape0
«
1gradients/encoder/conv2d/LeakyRelu/mul_grad/mul_1Mulencoder/conv2d/LeakyRelu/alphaHgradients/encoder/conv2d/LeakyRelu/Maximum_grad/tuple/control_dependency*
T0
Ö
1gradients/encoder/conv2d/LeakyRelu/mul_grad/Sum_1Sum1gradients/encoder/conv2d/LeakyRelu/mul_grad/mul_1Cgradients/encoder/conv2d/LeakyRelu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
¿
5gradients/encoder/conv2d/LeakyRelu/mul_grad/Reshape_1Reshape1gradients/encoder/conv2d/LeakyRelu/mul_grad/Sum_13gradients/encoder/conv2d/LeakyRelu/mul_grad/Shape_1*
T0*
Tshape0
²
<gradients/encoder/conv2d/LeakyRelu/mul_grad/tuple/group_depsNoOp4^gradients/encoder/conv2d/LeakyRelu/mul_grad/Reshape6^gradients/encoder/conv2d/LeakyRelu/mul_grad/Reshape_1

Dgradients/encoder/conv2d/LeakyRelu/mul_grad/tuple/control_dependencyIdentity3gradients/encoder/conv2d/LeakyRelu/mul_grad/Reshape=^gradients/encoder/conv2d/LeakyRelu/mul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/encoder/conv2d/LeakyRelu/mul_grad/Reshape

Fgradients/encoder/conv2d/LeakyRelu/mul_grad/tuple/control_dependency_1Identity5gradients/encoder/conv2d/LeakyRelu/mul_grad/Reshape_1=^gradients/encoder/conv2d/LeakyRelu/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/encoder/conv2d/LeakyRelu/mul_grad/Reshape_1

gradients/AddN_7AddNJgradients/encoder/conv2d/LeakyRelu/Maximum_grad/tuple/control_dependency_1Fgradients/encoder/conv2d/LeakyRelu/mul_grad/tuple/control_dependency_1*
T0*L
_classB
@>loc:@gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Reshape_1*
N
r
1gradients/encoder/conv2d/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_7*
T0*
data_formatNHWC

6gradients/encoder/conv2d/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_72^gradients/encoder/conv2d/BiasAdd_grad/BiasAddGrad
ì
>gradients/encoder/conv2d/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_77^gradients/encoder/conv2d/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/encoder/conv2d/LeakyRelu/Maximum_grad/Reshape_1

@gradients/encoder/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity1gradients/encoder/conv2d/BiasAdd_grad/BiasAddGrad7^gradients/encoder/conv2d/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/encoder/conv2d/BiasAdd_grad/BiasAddGrad

+gradients/encoder/conv2d/Conv2D_grad/ShapeNShapeNencoder/Reshapeencoder/conv2d/kernel/read*
out_type0*
N*
T0
¸
8gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput+gradients/encoder/conv2d/Conv2D_grad/ShapeNencoder/conv2d/kernel/read>gradients/encoder/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
T0
±
9gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterencoder/Reshape-gradients/encoder/conv2d/Conv2D_grad/ShapeN:1>gradients/encoder/conv2d/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
´
5gradients/encoder/conv2d/Conv2D_grad/tuple/group_depsNoOp9^gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropInput:^gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropFilter

=gradients/encoder/conv2d/Conv2D_grad/tuple/control_dependencyIdentity8gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropInput6^gradients/encoder/conv2d/Conv2D_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropInput

?gradients/encoder/conv2d/Conv2D_grad/tuple/control_dependency_1Identity9gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropFilter6^gradients/encoder/conv2d/Conv2D_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/encoder/conv2d/Conv2D_grad/Conv2DBackpropFilter
x
beta1_power/initial_valueConst*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
valueB
 *fff?*
dtype0

beta1_power
VariableV2*
dtype0*
	container *
shape: *
shared_name *0
_class&
$"loc:@decoder/conv2d_transpose/bias
¨
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
use_locking(*
T0*0
_class&
$"loc:@decoder/conv2d_transpose/bias
d
beta1_power/readIdentitybeta1_power*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
T0
x
beta2_power/initial_valueConst*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
valueB
 *w¾?*
dtype0

beta2_power
VariableV2*
	container *
shape: *
shared_name *0
_class&
$"loc:@decoder/conv2d_transpose/bias*
dtype0
¨
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
validate_shape(
d
beta2_power/readIdentitybeta2_power*
T0*0
_class&
$"loc:@decoder/conv2d_transpose/bias

,encoder/conv2d/kernel/Adam/Initializer/zerosConst*%
valueB@*    *(
_class
loc:@encoder/conv2d/kernel*
dtype0
 
encoder/conv2d/kernel/Adam
VariableV2*(
_class
loc:@encoder/conv2d/kernel*
dtype0*
	container *
shape:@*
shared_name 
Ñ
!encoder/conv2d/kernel/Adam/AssignAssignencoder/conv2d/kernel/Adam,encoder/conv2d/kernel/Adam/Initializer/zeros*(
_class
loc:@encoder/conv2d/kernel*
validate_shape(*
use_locking(*
T0
z
encoder/conv2d/kernel/Adam/readIdentityencoder/conv2d/kernel/Adam*
T0*(
_class
loc:@encoder/conv2d/kernel

.encoder/conv2d/kernel/Adam_1/Initializer/zerosConst*%
valueB@*    *(
_class
loc:@encoder/conv2d/kernel*
dtype0
¢
encoder/conv2d/kernel/Adam_1
VariableV2*(
_class
loc:@encoder/conv2d/kernel*
dtype0*
	container *
shape:@*
shared_name 
×
#encoder/conv2d/kernel/Adam_1/AssignAssignencoder/conv2d/kernel/Adam_1.encoder/conv2d/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@encoder/conv2d/kernel*
validate_shape(
~
!encoder/conv2d/kernel/Adam_1/readIdentityencoder/conv2d/kernel/Adam_1*
T0*(
_class
loc:@encoder/conv2d/kernel

*encoder/conv2d/bias/Adam/Initializer/zerosConst*
valueB@*    *&
_class
loc:@encoder/conv2d/bias*
dtype0

encoder/conv2d/bias/Adam
VariableV2*
shape:@*
shared_name *&
_class
loc:@encoder/conv2d/bias*
dtype0*
	container 
É
encoder/conv2d/bias/Adam/AssignAssignencoder/conv2d/bias/Adam*encoder/conv2d/bias/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@encoder/conv2d/bias*
validate_shape(
t
encoder/conv2d/bias/Adam/readIdentityencoder/conv2d/bias/Adam*
T0*&
_class
loc:@encoder/conv2d/bias

,encoder/conv2d/bias/Adam_1/Initializer/zerosConst*
valueB@*    *&
_class
loc:@encoder/conv2d/bias*
dtype0

encoder/conv2d/bias/Adam_1
VariableV2*&
_class
loc:@encoder/conv2d/bias*
dtype0*
	container *
shape:@*
shared_name 
Ï
!encoder/conv2d/bias/Adam_1/AssignAssignencoder/conv2d/bias/Adam_1,encoder/conv2d/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@encoder/conv2d/bias*
validate_shape(
x
encoder/conv2d/bias/Adam_1/readIdentityencoder/conv2d/bias/Adam_1*
T0*&
_class
loc:@encoder/conv2d/bias

.encoder/conv2d_1/kernel/Adam/Initializer/zerosConst*%
valueB@@*    **
_class 
loc:@encoder/conv2d_1/kernel*
dtype0
¤
encoder/conv2d_1/kernel/Adam
VariableV2**
_class 
loc:@encoder/conv2d_1/kernel*
dtype0*
	container *
shape:@@*
shared_name 
Ù
#encoder/conv2d_1/kernel/Adam/AssignAssignencoder/conv2d_1/kernel/Adam.encoder/conv2d_1/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0**
_class 
loc:@encoder/conv2d_1/kernel

!encoder/conv2d_1/kernel/Adam/readIdentityencoder/conv2d_1/kernel/Adam*
T0**
_class 
loc:@encoder/conv2d_1/kernel

0encoder/conv2d_1/kernel/Adam_1/Initializer/zerosConst*%
valueB@@*    **
_class 
loc:@encoder/conv2d_1/kernel*
dtype0
¦
encoder/conv2d_1/kernel/Adam_1
VariableV2*
	container *
shape:@@*
shared_name **
_class 
loc:@encoder/conv2d_1/kernel*
dtype0
ß
%encoder/conv2d_1/kernel/Adam_1/AssignAssignencoder/conv2d_1/kernel/Adam_10encoder/conv2d_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0**
_class 
loc:@encoder/conv2d_1/kernel

#encoder/conv2d_1/kernel/Adam_1/readIdentityencoder/conv2d_1/kernel/Adam_1*
T0**
_class 
loc:@encoder/conv2d_1/kernel

,encoder/conv2d_1/bias/Adam/Initializer/zerosConst*
valueB@*    *(
_class
loc:@encoder/conv2d_1/bias*
dtype0

encoder/conv2d_1/bias/Adam
VariableV2*
shape:@*
shared_name *(
_class
loc:@encoder/conv2d_1/bias*
dtype0*
	container 
Ñ
!encoder/conv2d_1/bias/Adam/AssignAssignencoder/conv2d_1/bias/Adam,encoder/conv2d_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@encoder/conv2d_1/bias*
validate_shape(
z
encoder/conv2d_1/bias/Adam/readIdentityencoder/conv2d_1/bias/Adam*
T0*(
_class
loc:@encoder/conv2d_1/bias

.encoder/conv2d_1/bias/Adam_1/Initializer/zerosConst*
valueB@*    *(
_class
loc:@encoder/conv2d_1/bias*
dtype0

encoder/conv2d_1/bias/Adam_1
VariableV2*
shape:@*
shared_name *(
_class
loc:@encoder/conv2d_1/bias*
dtype0*
	container 
×
#encoder/conv2d_1/bias/Adam_1/AssignAssignencoder/conv2d_1/bias/Adam_1.encoder/conv2d_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@encoder/conv2d_1/bias*
validate_shape(
~
!encoder/conv2d_1/bias/Adam_1/readIdentityencoder/conv2d_1/bias/Adam_1*(
_class
loc:@encoder/conv2d_1/bias*
T0

.encoder/conv2d_2/kernel/Adam/Initializer/zerosConst*
dtype0*%
valueB@@*    **
_class 
loc:@encoder/conv2d_2/kernel
¤
encoder/conv2d_2/kernel/Adam
VariableV2*
shape:@@*
shared_name **
_class 
loc:@encoder/conv2d_2/kernel*
dtype0*
	container 
Ù
#encoder/conv2d_2/kernel/Adam/AssignAssignencoder/conv2d_2/kernel/Adam.encoder/conv2d_2/kernel/Adam/Initializer/zeros**
_class 
loc:@encoder/conv2d_2/kernel*
validate_shape(*
use_locking(*
T0

!encoder/conv2d_2/kernel/Adam/readIdentityencoder/conv2d_2/kernel/Adam*
T0**
_class 
loc:@encoder/conv2d_2/kernel

0encoder/conv2d_2/kernel/Adam_1/Initializer/zerosConst*%
valueB@@*    **
_class 
loc:@encoder/conv2d_2/kernel*
dtype0
¦
encoder/conv2d_2/kernel/Adam_1
VariableV2**
_class 
loc:@encoder/conv2d_2/kernel*
dtype0*
	container *
shape:@@*
shared_name 
ß
%encoder/conv2d_2/kernel/Adam_1/AssignAssignencoder/conv2d_2/kernel/Adam_10encoder/conv2d_2/kernel/Adam_1/Initializer/zeros*
T0**
_class 
loc:@encoder/conv2d_2/kernel*
validate_shape(*
use_locking(

#encoder/conv2d_2/kernel/Adam_1/readIdentityencoder/conv2d_2/kernel/Adam_1*
T0**
_class 
loc:@encoder/conv2d_2/kernel

,encoder/conv2d_2/bias/Adam/Initializer/zerosConst*
valueB@*    *(
_class
loc:@encoder/conv2d_2/bias*
dtype0

encoder/conv2d_2/bias/Adam
VariableV2*(
_class
loc:@encoder/conv2d_2/bias*
dtype0*
	container *
shape:@*
shared_name 
Ñ
!encoder/conv2d_2/bias/Adam/AssignAssignencoder/conv2d_2/bias/Adam,encoder/conv2d_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@encoder/conv2d_2/bias*
validate_shape(
z
encoder/conv2d_2/bias/Adam/readIdentityencoder/conv2d_2/bias/Adam*
T0*(
_class
loc:@encoder/conv2d_2/bias

.encoder/conv2d_2/bias/Adam_1/Initializer/zerosConst*
valueB@*    *(
_class
loc:@encoder/conv2d_2/bias*
dtype0

encoder/conv2d_2/bias/Adam_1
VariableV2*(
_class
loc:@encoder/conv2d_2/bias*
dtype0*
	container *
shape:@*
shared_name 
×
#encoder/conv2d_2/bias/Adam_1/AssignAssignencoder/conv2d_2/bias/Adam_1.encoder/conv2d_2/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*(
_class
loc:@encoder/conv2d_2/bias
~
!encoder/conv2d_2/bias/Adam_1/readIdentityencoder/conv2d_2/bias/Adam_1*(
_class
loc:@encoder/conv2d_2/bias*
T0

+encoder/dense/kernel/Adam/Initializer/zerosConst*
dtype0*
valueB	À*    *'
_class
loc:@encoder/dense/kernel

encoder/dense/kernel/Adam
VariableV2*
shape:	À*
shared_name *'
_class
loc:@encoder/dense/kernel*
dtype0*
	container 
Í
 encoder/dense/kernel/Adam/AssignAssignencoder/dense/kernel/Adam+encoder/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*'
_class
loc:@encoder/dense/kernel
w
encoder/dense/kernel/Adam/readIdentityencoder/dense/kernel/Adam*
T0*'
_class
loc:@encoder/dense/kernel

-encoder/dense/kernel/Adam_1/Initializer/zerosConst*
valueB	À*    *'
_class
loc:@encoder/dense/kernel*
dtype0

encoder/dense/kernel/Adam_1
VariableV2*
shared_name *'
_class
loc:@encoder/dense/kernel*
dtype0*
	container *
shape:	À
Ó
"encoder/dense/kernel/Adam_1/AssignAssignencoder/dense/kernel/Adam_1-encoder/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@encoder/dense/kernel*
validate_shape(
{
 encoder/dense/kernel/Adam_1/readIdentityencoder/dense/kernel/Adam_1*
T0*'
_class
loc:@encoder/dense/kernel

)encoder/dense/bias/Adam/Initializer/zerosConst*
valueB*    *%
_class
loc:@encoder/dense/bias*
dtype0

encoder/dense/bias/Adam
VariableV2*%
_class
loc:@encoder/dense/bias*
dtype0*
	container *
shape:*
shared_name 
Å
encoder/dense/bias/Adam/AssignAssignencoder/dense/bias/Adam)encoder/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@encoder/dense/bias*
validate_shape(
q
encoder/dense/bias/Adam/readIdentityencoder/dense/bias/Adam*
T0*%
_class
loc:@encoder/dense/bias

+encoder/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *%
_class
loc:@encoder/dense/bias

encoder/dense/bias/Adam_1
VariableV2*
dtype0*
	container *
shape:*
shared_name *%
_class
loc:@encoder/dense/bias
Ë
 encoder/dense/bias/Adam_1/AssignAssignencoder/dense/bias/Adam_1+encoder/dense/bias/Adam_1/Initializer/zeros*%
_class
loc:@encoder/dense/bias*
validate_shape(*
use_locking(*
T0
u
encoder/dense/bias/Adam_1/readIdentityencoder/dense/bias/Adam_1*%
_class
loc:@encoder/dense/bias*
T0

-encoder/dense_1/kernel/Adam/Initializer/zerosConst*
valueB	À*    *)
_class
loc:@encoder/dense_1/kernel*
dtype0

encoder/dense_1/kernel/Adam
VariableV2*
shared_name *)
_class
loc:@encoder/dense_1/kernel*
dtype0*
	container *
shape:	À
Õ
"encoder/dense_1/kernel/Adam/AssignAssignencoder/dense_1/kernel/Adam-encoder/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@encoder/dense_1/kernel*
validate_shape(
}
 encoder/dense_1/kernel/Adam/readIdentityencoder/dense_1/kernel/Adam*
T0*)
_class
loc:@encoder/dense_1/kernel

/encoder/dense_1/kernel/Adam_1/Initializer/zerosConst*
valueB	À*    *)
_class
loc:@encoder/dense_1/kernel*
dtype0

encoder/dense_1/kernel/Adam_1
VariableV2*
dtype0*
	container *
shape:	À*
shared_name *)
_class
loc:@encoder/dense_1/kernel
Û
$encoder/dense_1/kernel/Adam_1/AssignAssignencoder/dense_1/kernel/Adam_1/encoder/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@encoder/dense_1/kernel*
validate_shape(

"encoder/dense_1/kernel/Adam_1/readIdentityencoder/dense_1/kernel/Adam_1*
T0*)
_class
loc:@encoder/dense_1/kernel

+encoder/dense_1/bias/Adam/Initializer/zerosConst*
valueB*    *'
_class
loc:@encoder/dense_1/bias*
dtype0

encoder/dense_1/bias/Adam
VariableV2*'
_class
loc:@encoder/dense_1/bias*
dtype0*
	container *
shape:*
shared_name 
Í
 encoder/dense_1/bias/Adam/AssignAssignencoder/dense_1/bias/Adam+encoder/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*'
_class
loc:@encoder/dense_1/bias
w
encoder/dense_1/bias/Adam/readIdentityencoder/dense_1/bias/Adam*
T0*'
_class
loc:@encoder/dense_1/bias

-encoder/dense_1/bias/Adam_1/Initializer/zerosConst*
valueB*    *'
_class
loc:@encoder/dense_1/bias*
dtype0

encoder/dense_1/bias/Adam_1
VariableV2*
shape:*
shared_name *'
_class
loc:@encoder/dense_1/bias*
dtype0*
	container 
Ó
"encoder/dense_1/bias/Adam_1/AssignAssignencoder/dense_1/bias/Adam_1-encoder/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@encoder/dense_1/bias*
validate_shape(
{
 encoder/dense_1/bias/Adam_1/readIdentityencoder/dense_1/bias/Adam_1*
T0*'
_class
loc:@encoder/dense_1/bias

+decoder/dense/kernel/Adam/Initializer/zerosConst*
dtype0*
valueB*    *'
_class
loc:@decoder/dense/kernel

decoder/dense/kernel/Adam
VariableV2*
dtype0*
	container *
shape
:*
shared_name *'
_class
loc:@decoder/dense/kernel
Í
 decoder/dense/kernel/Adam/AssignAssigndecoder/dense/kernel/Adam+decoder/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@decoder/dense/kernel*
validate_shape(
w
decoder/dense/kernel/Adam/readIdentitydecoder/dense/kernel/Adam*
T0*'
_class
loc:@decoder/dense/kernel

-decoder/dense/kernel/Adam_1/Initializer/zerosConst*
valueB*    *'
_class
loc:@decoder/dense/kernel*
dtype0

decoder/dense/kernel/Adam_1
VariableV2*
shape
:*
shared_name *'
_class
loc:@decoder/dense/kernel*
dtype0*
	container 
Ó
"decoder/dense/kernel/Adam_1/AssignAssigndecoder/dense/kernel/Adam_1-decoder/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@decoder/dense/kernel*
validate_shape(
{
 decoder/dense/kernel/Adam_1/readIdentitydecoder/dense/kernel/Adam_1*
T0*'
_class
loc:@decoder/dense/kernel

)decoder/dense/bias/Adam/Initializer/zerosConst*
valueB*    *%
_class
loc:@decoder/dense/bias*
dtype0

decoder/dense/bias/Adam
VariableV2*
dtype0*
	container *
shape:*
shared_name *%
_class
loc:@decoder/dense/bias
Å
decoder/dense/bias/Adam/AssignAssigndecoder/dense/bias/Adam)decoder/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@decoder/dense/bias*
validate_shape(
q
decoder/dense/bias/Adam/readIdentitydecoder/dense/bias/Adam*
T0*%
_class
loc:@decoder/dense/bias

+decoder/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *%
_class
loc:@decoder/dense/bias*
dtype0

decoder/dense/bias/Adam_1
VariableV2*%
_class
loc:@decoder/dense/bias*
dtype0*
	container *
shape:*
shared_name 
Ë
 decoder/dense/bias/Adam_1/AssignAssigndecoder/dense/bias/Adam_1+decoder/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@decoder/dense/bias*
validate_shape(
u
decoder/dense/bias/Adam_1/readIdentitydecoder/dense/bias/Adam_1*
T0*%
_class
loc:@decoder/dense/bias

-decoder/dense_1/kernel/Adam/Initializer/zerosConst*
valueB1*    *)
_class
loc:@decoder/dense_1/kernel*
dtype0

decoder/dense_1/kernel/Adam
VariableV2*)
_class
loc:@decoder/dense_1/kernel*
dtype0*
	container *
shape
:1*
shared_name 
Õ
"decoder/dense_1/kernel/Adam/AssignAssigndecoder/dense_1/kernel/Adam-decoder/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@decoder/dense_1/kernel*
validate_shape(
}
 decoder/dense_1/kernel/Adam/readIdentitydecoder/dense_1/kernel/Adam*
T0*)
_class
loc:@decoder/dense_1/kernel

/decoder/dense_1/kernel/Adam_1/Initializer/zerosConst*
valueB1*    *)
_class
loc:@decoder/dense_1/kernel*
dtype0

decoder/dense_1/kernel/Adam_1
VariableV2*
dtype0*
	container *
shape
:1*
shared_name *)
_class
loc:@decoder/dense_1/kernel
Û
$decoder/dense_1/kernel/Adam_1/AssignAssigndecoder/dense_1/kernel/Adam_1/decoder/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@decoder/dense_1/kernel*
validate_shape(

"decoder/dense_1/kernel/Adam_1/readIdentitydecoder/dense_1/kernel/Adam_1*
T0*)
_class
loc:@decoder/dense_1/kernel

+decoder/dense_1/bias/Adam/Initializer/zerosConst*
dtype0*
valueB1*    *'
_class
loc:@decoder/dense_1/bias

decoder/dense_1/bias/Adam
VariableV2*
dtype0*
	container *
shape:1*
shared_name *'
_class
loc:@decoder/dense_1/bias
Í
 decoder/dense_1/bias/Adam/AssignAssigndecoder/dense_1/bias/Adam+decoder/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*'
_class
loc:@decoder/dense_1/bias
w
decoder/dense_1/bias/Adam/readIdentitydecoder/dense_1/bias/Adam*
T0*'
_class
loc:@decoder/dense_1/bias

-decoder/dense_1/bias/Adam_1/Initializer/zerosConst*
valueB1*    *'
_class
loc:@decoder/dense_1/bias*
dtype0

decoder/dense_1/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@decoder/dense_1/bias*
dtype0*
	container *
shape:1
Ó
"decoder/dense_1/bias/Adam_1/AssignAssigndecoder/dense_1/bias/Adam_1-decoder/dense_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*'
_class
loc:@decoder/dense_1/bias
{
 decoder/dense_1/bias/Adam_1/readIdentitydecoder/dense_1/bias/Adam_1*
T0*'
_class
loc:@decoder/dense_1/bias
§
6decoder/conv2d_transpose/kernel/Adam/Initializer/zerosConst*%
valueB@*    *2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0
´
$decoder/conv2d_transpose/kernel/Adam
VariableV2*
dtype0*
	container *
shape:@*
shared_name *2
_class(
&$loc:@decoder/conv2d_transpose/kernel
ù
+decoder/conv2d_transpose/kernel/Adam/AssignAssign$decoder/conv2d_transpose/kernel/Adam6decoder/conv2d_transpose/kernel/Adam/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
validate_shape(

)decoder/conv2d_transpose/kernel/Adam/readIdentity$decoder/conv2d_transpose/kernel/Adam*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel
©
8decoder/conv2d_transpose/kernel/Adam_1/Initializer/zerosConst*%
valueB@*    *2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0
¶
&decoder/conv2d_transpose/kernel/Adam_1
VariableV2*
	container *
shape:@*
shared_name *2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
dtype0
ÿ
-decoder/conv2d_transpose/kernel/Adam_1/AssignAssign&decoder/conv2d_transpose/kernel/Adam_18decoder/conv2d_transpose/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
validate_shape(

+decoder/conv2d_transpose/kernel/Adam_1/readIdentity&decoder/conv2d_transpose/kernel/Adam_1*
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel

4decoder/conv2d_transpose/bias/Adam/Initializer/zerosConst*
valueB@*    *0
_class&
$"loc:@decoder/conv2d_transpose/bias*
dtype0
¤
"decoder/conv2d_transpose/bias/Adam
VariableV2*
	container *
shape:@*
shared_name *0
_class&
$"loc:@decoder/conv2d_transpose/bias*
dtype0
ñ
)decoder/conv2d_transpose/bias/Adam/AssignAssign"decoder/conv2d_transpose/bias/Adam4decoder/conv2d_transpose/bias/Adam/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
validate_shape(

'decoder/conv2d_transpose/bias/Adam/readIdentity"decoder/conv2d_transpose/bias/Adam*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
T0

6decoder/conv2d_transpose/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB@*    *0
_class&
$"loc:@decoder/conv2d_transpose/bias
¦
$decoder/conv2d_transpose/bias/Adam_1
VariableV2*
dtype0*
	container *
shape:@*
shared_name *0
_class&
$"loc:@decoder/conv2d_transpose/bias
÷
+decoder/conv2d_transpose/bias/Adam_1/AssignAssign$decoder/conv2d_transpose/bias/Adam_16decoder/conv2d_transpose/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*0
_class&
$"loc:@decoder/conv2d_transpose/bias

)decoder/conv2d_transpose/bias/Adam_1/readIdentity$decoder/conv2d_transpose/bias/Adam_1*
T0*0
_class&
$"loc:@decoder/conv2d_transpose/bias
«
8decoder/conv2d_transpose_1/kernel/Adam/Initializer/zerosConst*%
valueB@@*    *4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0
¸
&decoder/conv2d_transpose_1/kernel/Adam
VariableV2*
shared_name *4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0*
	container *
shape:@@

-decoder/conv2d_transpose_1/kernel/Adam/AssignAssign&decoder/conv2d_transpose_1/kernel/Adam8decoder/conv2d_transpose_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
validate_shape(

+decoder/conv2d_transpose_1/kernel/Adam/readIdentity&decoder/conv2d_transpose_1/kernel/Adam*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel
­
:decoder/conv2d_transpose_1/kernel/Adam_1/Initializer/zerosConst*%
valueB@@*    *4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0
º
(decoder/conv2d_transpose_1/kernel/Adam_1
VariableV2*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
dtype0*
	container *
shape:@@*
shared_name 

/decoder/conv2d_transpose_1/kernel/Adam_1/AssignAssign(decoder/conv2d_transpose_1/kernel/Adam_1:decoder/conv2d_transpose_1/kernel/Adam_1/Initializer/zeros*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
validate_shape(*
use_locking(
¢
-decoder/conv2d_transpose_1/kernel/Adam_1/readIdentity(decoder/conv2d_transpose_1/kernel/Adam_1*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel

6decoder/conv2d_transpose_1/bias/Adam/Initializer/zerosConst*
dtype0*
valueB@*    *2
_class(
&$loc:@decoder/conv2d_transpose_1/bias
¨
$decoder/conv2d_transpose_1/bias/Adam
VariableV2*
shared_name *2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
dtype0*
	container *
shape:@
ù
+decoder/conv2d_transpose_1/bias/Adam/AssignAssign$decoder/conv2d_transpose_1/bias/Adam6decoder/conv2d_transpose_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
validate_shape(

)decoder/conv2d_transpose_1/bias/Adam/readIdentity$decoder/conv2d_transpose_1/bias/Adam*
T0*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias

8decoder/conv2d_transpose_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB@*    *2
_class(
&$loc:@decoder/conv2d_transpose_1/bias
ª
&decoder/conv2d_transpose_1/bias/Adam_1
VariableV2*
	container *
shape:@*
shared_name *2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
dtype0
ÿ
-decoder/conv2d_transpose_1/bias/Adam_1/AssignAssign&decoder/conv2d_transpose_1/bias/Adam_18decoder/conv2d_transpose_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias

+decoder/conv2d_transpose_1/bias/Adam_1/readIdentity&decoder/conv2d_transpose_1/bias/Adam_1*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
T0
«
8decoder/conv2d_transpose_2/kernel/Adam/Initializer/zerosConst*%
valueB@@*    *4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0
¸
&decoder/conv2d_transpose_2/kernel/Adam
VariableV2*
shape:@@*
shared_name *4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0*
	container 

-decoder/conv2d_transpose_2/kernel/Adam/AssignAssign&decoder/conv2d_transpose_2/kernel/Adam8decoder/conv2d_transpose_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
validate_shape(

+decoder/conv2d_transpose_2/kernel/Adam/readIdentity&decoder/conv2d_transpose_2/kernel/Adam*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel
­
:decoder/conv2d_transpose_2/kernel/Adam_1/Initializer/zerosConst*%
valueB@@*    *4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
dtype0
º
(decoder/conv2d_transpose_2/kernel/Adam_1
VariableV2*
dtype0*
	container *
shape:@@*
shared_name *4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel

/decoder/conv2d_transpose_2/kernel/Adam_1/AssignAssign(decoder/conv2d_transpose_2/kernel/Adam_1:decoder/conv2d_transpose_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel*
validate_shape(
¢
-decoder/conv2d_transpose_2/kernel/Adam_1/readIdentity(decoder/conv2d_transpose_2/kernel/Adam_1*
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel

6decoder/conv2d_transpose_2/bias/Adam/Initializer/zerosConst*
valueB@*    *2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
dtype0
¨
$decoder/conv2d_transpose_2/bias/Adam
VariableV2*
shape:@*
shared_name *2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
dtype0*
	container 
ù
+decoder/conv2d_transpose_2/bias/Adam/AssignAssign$decoder/conv2d_transpose_2/bias/Adam6decoder/conv2d_transpose_2/bias/Adam/Initializer/zeros*
T0*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
validate_shape(*
use_locking(

)decoder/conv2d_transpose_2/bias/Adam/readIdentity$decoder/conv2d_transpose_2/bias/Adam*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
T0

8decoder/conv2d_transpose_2/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB@*    *2
_class(
&$loc:@decoder/conv2d_transpose_2/bias
ª
&decoder/conv2d_transpose_2/bias/Adam_1
VariableV2*
shared_name *2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
dtype0*
	container *
shape:@
ÿ
-decoder/conv2d_transpose_2/bias/Adam_1/AssignAssign&decoder/conv2d_transpose_2/bias/Adam_18decoder/conv2d_transpose_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
validate_shape(

+decoder/conv2d_transpose_2/bias/Adam_1/readIdentity&decoder/conv2d_transpose_2/bias/Adam_1*
T0*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias

-decoder/dense_2/kernel/Adam/Initializer/zerosConst*
valueB
b*    *)
_class
loc:@decoder/dense_2/kernel*
dtype0

decoder/dense_2/kernel/Adam
VariableV2*
shared_name *)
_class
loc:@decoder/dense_2/kernel*
dtype0*
	container *
shape:
b
Õ
"decoder/dense_2/kernel/Adam/AssignAssigndecoder/dense_2/kernel/Adam-decoder/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@decoder/dense_2/kernel*
validate_shape(
}
 decoder/dense_2/kernel/Adam/readIdentitydecoder/dense_2/kernel/Adam*
T0*)
_class
loc:@decoder/dense_2/kernel

/decoder/dense_2/kernel/Adam_1/Initializer/zerosConst*
valueB
b*    *)
_class
loc:@decoder/dense_2/kernel*
dtype0

decoder/dense_2/kernel/Adam_1
VariableV2*
shape:
b*
shared_name *)
_class
loc:@decoder/dense_2/kernel*
dtype0*
	container 
Û
$decoder/dense_2/kernel/Adam_1/AssignAssigndecoder/dense_2/kernel/Adam_1/decoder/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*)
_class
loc:@decoder/dense_2/kernel

"decoder/dense_2/kernel/Adam_1/readIdentitydecoder/dense_2/kernel/Adam_1*
T0*)
_class
loc:@decoder/dense_2/kernel

+decoder/dense_2/bias/Adam/Initializer/zerosConst*
valueB*    *'
_class
loc:@decoder/dense_2/bias*
dtype0

decoder/dense_2/bias/Adam
VariableV2*
shared_name *'
_class
loc:@decoder/dense_2/bias*
dtype0*
	container *
shape:
Í
 decoder/dense_2/bias/Adam/AssignAssigndecoder/dense_2/bias/Adam+decoder/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@decoder/dense_2/bias*
validate_shape(
w
decoder/dense_2/bias/Adam/readIdentitydecoder/dense_2/bias/Adam*
T0*'
_class
loc:@decoder/dense_2/bias

-decoder/dense_2/bias/Adam_1/Initializer/zerosConst*
valueB*    *'
_class
loc:@decoder/dense_2/bias*
dtype0

decoder/dense_2/bias/Adam_1
VariableV2*
shape:*
shared_name *'
_class
loc:@decoder/dense_2/bias*
dtype0*
	container 
Ó
"decoder/dense_2/bias/Adam_1/AssignAssigndecoder/dense_2/bias/Adam_1-decoder/dense_2/bias/Adam_1/Initializer/zeros*'
_class
loc:@decoder/dense_2/bias*
validate_shape(*
use_locking(*
T0
{
 decoder/dense_2/bias/Adam_1/readIdentitydecoder/dense_2/bias/Adam_1*
T0*'
_class
loc:@decoder/dense_2/bias
?
Adam/learning_rateConst*
dtype0*
valueB
 *o:
7

Adam/beta1Const*
valueB
 *fff?*
dtype0
7

Adam/beta2Const*
valueB
 *w¾?*
dtype0
9
Adam/epsilonConst*
valueB
 *wÌ+2*
dtype0

+Adam/update_encoder/conv2d/kernel/ApplyAdam	ApplyAdamencoder/conv2d/kernelencoder/conv2d/kernel/Adamencoder/conv2d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/encoder/conv2d/Conv2D_grad/tuple/control_dependency_1*
T0*(
_class
loc:@encoder/conv2d/kernel*
use_nesterov( *
use_locking( 
ù
)Adam/update_encoder/conv2d/bias/ApplyAdam	ApplyAdamencoder/conv2d/biasencoder/conv2d/bias/Adamencoder/conv2d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/encoder/conv2d/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@encoder/conv2d/bias*
use_nesterov( 

-Adam/update_encoder/conv2d_1/kernel/ApplyAdam	ApplyAdamencoder/conv2d_1/kernelencoder/conv2d_1/kernel/Adamencoder/conv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/encoder/conv2d_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@encoder/conv2d_1/kernel*
use_nesterov( 

+Adam/update_encoder/conv2d_1/bias/ApplyAdam	ApplyAdamencoder/conv2d_1/biasencoder/conv2d_1/bias/Adamencoder/conv2d_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/encoder/conv2d_2/BiasAdd_grad/tuple/control_dependency_1*
T0*(
_class
loc:@encoder/conv2d_1/bias*
use_nesterov( *
use_locking( 

-Adam/update_encoder/conv2d_2/kernel/ApplyAdam	ApplyAdamencoder/conv2d_2/kernelencoder/conv2d_2/kernel/Adamencoder/conv2d_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/encoder/conv2d_3/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0**
_class 
loc:@encoder/conv2d_2/kernel

+Adam/update_encoder/conv2d_2/bias/ApplyAdam	ApplyAdamencoder/conv2d_2/biasencoder/conv2d_2/bias/Adamencoder/conv2d_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/encoder/conv2d_3/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*(
_class
loc:@encoder/conv2d_2/bias
ü
*Adam/update_encoder/dense/kernel/ApplyAdam	ApplyAdamencoder/dense/kernelencoder/dense/kernel/Adamencoder/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/encoder/dense/MatMul_grad/tuple/control_dependency_1*
T0*'
_class
loc:@encoder/dense/kernel*
use_nesterov( *
use_locking( 
ó
(Adam/update_encoder/dense/bias/ApplyAdam	ApplyAdamencoder/dense/biasencoder/dense/bias/Adamencoder/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/encoder/dense/BiasAdd_grad/tuple/control_dependency_1*%
_class
loc:@encoder/dense/bias*
use_nesterov( *
use_locking( *
T0

,Adam/update_encoder/dense_1/kernel/ApplyAdam	ApplyAdamencoder/dense_1/kernelencoder/dense_1/kernel/Adamencoder/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/encoder/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@encoder/dense_1/kernel*
use_nesterov( 
ÿ
*Adam/update_encoder/dense_1/bias/ApplyAdam	ApplyAdamencoder/dense_1/biasencoder/dense_1/bias/Adamencoder/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/encoder/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@encoder/dense_1/bias*
use_nesterov( 
ü
*Adam/update_decoder/dense/kernel/ApplyAdam	ApplyAdamdecoder/dense/kerneldecoder/dense/kernel/Adamdecoder/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/decoder/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@decoder/dense/kernel*
use_nesterov( 
ó
(Adam/update_decoder/dense/bias/ApplyAdam	ApplyAdamdecoder/dense/biasdecoder/dense/bias/Adamdecoder/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/decoder/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@decoder/dense/bias*
use_nesterov( 

,Adam/update_decoder/dense_1/kernel/ApplyAdam	ApplyAdamdecoder/dense_1/kerneldecoder/dense_1/kernel/Adamdecoder/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/decoder/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@decoder/dense_1/kernel*
use_nesterov( 
ÿ
*Adam/update_decoder/dense_1/bias/ApplyAdam	ApplyAdamdecoder/dense_1/biasdecoder/dense_1/bias/Adamdecoder/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/decoder/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@decoder/dense_1/bias*
use_nesterov( 
È
5Adam/update_decoder/conv2d_transpose/kernel/ApplyAdam	ApplyAdamdecoder/conv2d_transpose/kernel$decoder/conv2d_transpose/kernel/Adam&decoder/conv2d_transpose/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonSgradients/decoder/conv2d_transpose/conv2d_transpose_2_grad/tuple/control_dependency*
use_locking( *
T0*2
_class(
&$loc:@decoder/conv2d_transpose/kernel*
use_nesterov( 
µ
3Adam/update_decoder/conv2d_transpose/bias/ApplyAdam	ApplyAdamdecoder/conv2d_transpose/bias"decoder/conv2d_transpose/bias/Adam$decoder/conv2d_transpose/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonJgradients/decoder/conv2d_transpose/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
use_nesterov( 
Ò
7Adam/update_decoder/conv2d_transpose_1/kernel/ApplyAdam	ApplyAdam!decoder/conv2d_transpose_1/kernel&decoder/conv2d_transpose_1/kernel/Adam(decoder/conv2d_transpose_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonSgradients/decoder/conv2d_transpose_2/conv2d_transpose_grad/tuple/control_dependency*
use_locking( *
T0*4
_class*
(&loc:@decoder/conv2d_transpose_1/kernel*
use_nesterov( 
Á
5Adam/update_decoder/conv2d_transpose_1/bias/ApplyAdam	ApplyAdamdecoder/conv2d_transpose_1/bias$decoder/conv2d_transpose_1/bias/Adam&decoder/conv2d_transpose_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/decoder/conv2d_transpose_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*2
_class(
&$loc:@decoder/conv2d_transpose_1/bias*
use_nesterov( 
Ò
7Adam/update_decoder/conv2d_transpose_2/kernel/ApplyAdam	ApplyAdam!decoder/conv2d_transpose_2/kernel&decoder/conv2d_transpose_2/kernel/Adam(decoder/conv2d_transpose_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonSgradients/decoder/conv2d_transpose_3/conv2d_transpose_grad/tuple/control_dependency*
use_nesterov( *
use_locking( *
T0*4
_class*
(&loc:@decoder/conv2d_transpose_2/kernel
Á
5Adam/update_decoder/conv2d_transpose_2/bias/ApplyAdam	ApplyAdamdecoder/conv2d_transpose_2/bias$decoder/conv2d_transpose_2/bias/Adam&decoder/conv2d_transpose_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/decoder/conv2d_transpose_3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*2
_class(
&$loc:@decoder/conv2d_transpose_2/bias*
use_nesterov( 

,Adam/update_decoder/dense_2/kernel/ApplyAdam	ApplyAdamdecoder/dense_2/kerneldecoder/dense_2/kernel/Adamdecoder/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/decoder/dense_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@decoder/dense_2/kernel*
use_nesterov( 
ÿ
*Adam/update_decoder/dense_2/bias/ApplyAdam	ApplyAdamdecoder/dense_2/biasdecoder/dense_2/bias/Adamdecoder/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/decoder/dense_3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@decoder/dense_2/bias*
use_nesterov( 
	
Adam/mulMulbeta1_power/read
Adam/beta1,^Adam/update_encoder/conv2d/kernel/ApplyAdam*^Adam/update_encoder/conv2d/bias/ApplyAdam.^Adam/update_encoder/conv2d_1/kernel/ApplyAdam,^Adam/update_encoder/conv2d_1/bias/ApplyAdam.^Adam/update_encoder/conv2d_2/kernel/ApplyAdam,^Adam/update_encoder/conv2d_2/bias/ApplyAdam+^Adam/update_encoder/dense/kernel/ApplyAdam)^Adam/update_encoder/dense/bias/ApplyAdam-^Adam/update_encoder/dense_1/kernel/ApplyAdam+^Adam/update_encoder/dense_1/bias/ApplyAdam+^Adam/update_decoder/dense/kernel/ApplyAdam)^Adam/update_decoder/dense/bias/ApplyAdam-^Adam/update_decoder/dense_1/kernel/ApplyAdam+^Adam/update_decoder/dense_1/bias/ApplyAdam6^Adam/update_decoder/conv2d_transpose/kernel/ApplyAdam4^Adam/update_decoder/conv2d_transpose/bias/ApplyAdam8^Adam/update_decoder/conv2d_transpose_1/kernel/ApplyAdam6^Adam/update_decoder/conv2d_transpose_1/bias/ApplyAdam8^Adam/update_decoder/conv2d_transpose_2/kernel/ApplyAdam6^Adam/update_decoder/conv2d_transpose_2/bias/ApplyAdam-^Adam/update_decoder/dense_2/kernel/ApplyAdam+^Adam/update_decoder/dense_2/bias/ApplyAdam*
T0*0
_class&
$"loc:@decoder/conv2d_transpose/bias

Adam/AssignAssignbeta1_powerAdam/mul*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
validate_shape(*
use_locking( *
T0
	

Adam/mul_1Mulbeta2_power/read
Adam/beta2,^Adam/update_encoder/conv2d/kernel/ApplyAdam*^Adam/update_encoder/conv2d/bias/ApplyAdam.^Adam/update_encoder/conv2d_1/kernel/ApplyAdam,^Adam/update_encoder/conv2d_1/bias/ApplyAdam.^Adam/update_encoder/conv2d_2/kernel/ApplyAdam,^Adam/update_encoder/conv2d_2/bias/ApplyAdam+^Adam/update_encoder/dense/kernel/ApplyAdam)^Adam/update_encoder/dense/bias/ApplyAdam-^Adam/update_encoder/dense_1/kernel/ApplyAdam+^Adam/update_encoder/dense_1/bias/ApplyAdam+^Adam/update_decoder/dense/kernel/ApplyAdam)^Adam/update_decoder/dense/bias/ApplyAdam-^Adam/update_decoder/dense_1/kernel/ApplyAdam+^Adam/update_decoder/dense_1/bias/ApplyAdam6^Adam/update_decoder/conv2d_transpose/kernel/ApplyAdam4^Adam/update_decoder/conv2d_transpose/bias/ApplyAdam8^Adam/update_decoder/conv2d_transpose_1/kernel/ApplyAdam6^Adam/update_decoder/conv2d_transpose_1/bias/ApplyAdam8^Adam/update_decoder/conv2d_transpose_2/kernel/ApplyAdam6^Adam/update_decoder/conv2d_transpose_2/bias/ApplyAdam-^Adam/update_decoder/dense_2/kernel/ApplyAdam+^Adam/update_decoder/dense_2/bias/ApplyAdam*
T0*0
_class&
$"loc:@decoder/conv2d_transpose/bias

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*0
_class&
$"loc:@decoder/conv2d_transpose/bias*
validate_shape(
Ö
AdamNoOp,^Adam/update_encoder/conv2d/kernel/ApplyAdam*^Adam/update_encoder/conv2d/bias/ApplyAdam.^Adam/update_encoder/conv2d_1/kernel/ApplyAdam,^Adam/update_encoder/conv2d_1/bias/ApplyAdam.^Adam/update_encoder/conv2d_2/kernel/ApplyAdam,^Adam/update_encoder/conv2d_2/bias/ApplyAdam+^Adam/update_encoder/dense/kernel/ApplyAdam)^Adam/update_encoder/dense/bias/ApplyAdam-^Adam/update_encoder/dense_1/kernel/ApplyAdam+^Adam/update_encoder/dense_1/bias/ApplyAdam+^Adam/update_decoder/dense/kernel/ApplyAdam)^Adam/update_decoder/dense/bias/ApplyAdam-^Adam/update_decoder/dense_1/kernel/ApplyAdam+^Adam/update_decoder/dense_1/bias/ApplyAdam6^Adam/update_decoder/conv2d_transpose/kernel/ApplyAdam4^Adam/update_decoder/conv2d_transpose/bias/ApplyAdam8^Adam/update_decoder/conv2d_transpose_1/kernel/ApplyAdam6^Adam/update_decoder/conv2d_transpose_1/bias/ApplyAdam8^Adam/update_decoder/conv2d_transpose_2/kernel/ApplyAdam6^Adam/update_decoder/conv2d_transpose_2/bias/ApplyAdam-^Adam/update_decoder/dense_2/kernel/ApplyAdam+^Adam/update_decoder/dense_2/bias/ApplyAdam^Adam/Assign^Adam/Assign_1
ä
initNoOp^encoder/conv2d/kernel/Assign^encoder/conv2d/bias/Assign^encoder/conv2d_1/kernel/Assign^encoder/conv2d_1/bias/Assign^encoder/conv2d_2/kernel/Assign^encoder/conv2d_2/bias/Assign^encoder/dense/kernel/Assign^encoder/dense/bias/Assign^encoder/dense_1/kernel/Assign^encoder/dense_1/bias/Assign^decoder/dense/kernel/Assign^decoder/dense/bias/Assign^decoder/dense_1/kernel/Assign^decoder/dense_1/bias/Assign'^decoder/conv2d_transpose/kernel/Assign%^decoder/conv2d_transpose/bias/Assign)^decoder/conv2d_transpose_1/kernel/Assign'^decoder/conv2d_transpose_1/bias/Assign)^decoder/conv2d_transpose_2/kernel/Assign'^decoder/conv2d_transpose_2/bias/Assign^decoder/dense_2/kernel/Assign^decoder/dense_2/bias/Assign^beta1_power/Assign^beta2_power/Assign"^encoder/conv2d/kernel/Adam/Assign$^encoder/conv2d/kernel/Adam_1/Assign ^encoder/conv2d/bias/Adam/Assign"^encoder/conv2d/bias/Adam_1/Assign$^encoder/conv2d_1/kernel/Adam/Assign&^encoder/conv2d_1/kernel/Adam_1/Assign"^encoder/conv2d_1/bias/Adam/Assign$^encoder/conv2d_1/bias/Adam_1/Assign$^encoder/conv2d_2/kernel/Adam/Assign&^encoder/conv2d_2/kernel/Adam_1/Assign"^encoder/conv2d_2/bias/Adam/Assign$^encoder/conv2d_2/bias/Adam_1/Assign!^encoder/dense/kernel/Adam/Assign#^encoder/dense/kernel/Adam_1/Assign^encoder/dense/bias/Adam/Assign!^encoder/dense/bias/Adam_1/Assign#^encoder/dense_1/kernel/Adam/Assign%^encoder/dense_1/kernel/Adam_1/Assign!^encoder/dense_1/bias/Adam/Assign#^encoder/dense_1/bias/Adam_1/Assign!^decoder/dense/kernel/Adam/Assign#^decoder/dense/kernel/Adam_1/Assign^decoder/dense/bias/Adam/Assign!^decoder/dense/bias/Adam_1/Assign#^decoder/dense_1/kernel/Adam/Assign%^decoder/dense_1/kernel/Adam_1/Assign!^decoder/dense_1/bias/Adam/Assign#^decoder/dense_1/bias/Adam_1/Assign,^decoder/conv2d_transpose/kernel/Adam/Assign.^decoder/conv2d_transpose/kernel/Adam_1/Assign*^decoder/conv2d_transpose/bias/Adam/Assign,^decoder/conv2d_transpose/bias/Adam_1/Assign.^decoder/conv2d_transpose_1/kernel/Adam/Assign0^decoder/conv2d_transpose_1/kernel/Adam_1/Assign,^decoder/conv2d_transpose_1/bias/Adam/Assign.^decoder/conv2d_transpose_1/bias/Adam_1/Assign.^decoder/conv2d_transpose_2/kernel/Adam/Assign0^decoder/conv2d_transpose_2/kernel/Adam_1/Assign,^decoder/conv2d_transpose_2/bias/Adam/Assign.^decoder/conv2d_transpose_2/bias/Adam_1/Assign#^decoder/dense_2/kernel/Adam/Assign%^decoder/dense_2/kernel/Adam_1/Assign!^decoder/dense_2/bias/Adam/Assign#^decoder/dense_2/bias/Adam_1/Assign"