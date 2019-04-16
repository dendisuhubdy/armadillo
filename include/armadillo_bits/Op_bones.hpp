// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


//! \addtogroup Op
//! @{



template<typename T1, typename op_type, bool condition>
struct Op_traits {};
  

template<typename T1, typename op_type>
struct Op_traits<T1, op_type, true>
  {
  static const bool is_row  = op_type::template traits<T1>::is_row;
  static const bool is_col  = op_type::template traits<T1>::is_col;
  static const bool is_xvec = op_type::template traits<T1>::is_xvec;
  };

template<typename T1, typename op_type>
struct Op_traits<T1, op_type, false>
  {
  static const bool is_row  = false;
  static const bool is_col  = false;
  static const bool is_xvec = false;
  };


template<typename T1, typename op_type>
class Op
  : public Base<typename T1::elem_type, Op<T1, op_type> >
  , public Op_traits<T1, op_type, has_nested_traits<op_type>::value >
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  inline explicit Op(const T1& in_m);
  inline          Op(const T1& in_m, const elem_type in_aux);
  inline          Op(const T1& in_m, const elem_type in_aux,         const uword in_aux_uword_a, const uword in_aux_uword_b);
  inline          Op(const T1& in_m, const uword     in_aux_uword_a, const uword in_aux_uword_b);
  inline          Op(const T1& in_m, const uword     in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c, const char junk);
  inline         ~Op();
  
  arma_aligned const T1&       m;            //!< the operand; must be derived from Base
  arma_aligned       elem_type aux;          //!< auxiliary data, using the element type as used by T1
  arma_aligned       uword     aux_uword_a;  //!< auxiliary data, uword format
  arma_aligned       uword     aux_uword_b;  //!< auxiliary data, uword format
  arma_aligned       uword     aux_uword_c;  //!< auxiliary data, uword format
  
  
  
  // static const bool is_row = \
  //   (
  //   // operations which result in a row vector if the input is a row vector
  //   T1::is_row &&
  //     (
  //        is_same_type<op_type, op_sort>::yes              DONE
  //     || is_same_type<op_type, op_sort_default>::yes      DONE
  //     || is_same_type<op_type, op_shift>::yes             DONE
  //     || is_same_type<op_type, op_shift_default>::yes     DONE
  //     || is_same_type<op_type, op_shuffle>::yes           DONE
  //     || is_same_type<op_type, op_shuffle_default>::yes   DONE
  //     || is_same_type<op_type, op_cumsum_default>::yes
  //     || is_same_type<op_type, op_cumprod_default>::yes
  //     || is_same_type<op_type, op_flipud>::yes
  //     || is_same_type<op_type, op_fliplr>::yes
  //     || is_same_type<op_type, op_reverse>::yes
  //     || is_same_type<op_type, op_reverse_default>::yes
  //     || is_same_type<op_type, op_unique>::yes
  //     || is_same_type<op_type, op_diff_default>::yes
  //     || is_same_type<op_type, op_normalise_vec>::yes
  //     || is_same_type<op_type, op_chi2rnd>::yes
  //     )
  //   )
  //   ||
  //   (
  //   // operations which result in a row vector if the input is a column vector
  //   T1::is_col &&
  //     (
  //        is_same_type<op_type, op_strans>::yes    DONE
  //     || is_same_type<op_type, op_htrans>::yes    DONE
  //     || is_same_type<op_type, op_htrans2>::yes   DONE
  //     )
  //   )
  //   ;
  // 
  // static const bool is_col = \
  //   (
  //   // operations which always result in a column vector
  //      is_same_type<op_type, op_diagvec>::yes             DONE
  //   || is_same_type<op_type, op_vectorise_col>::yes       DONE
  //   || is_same_type<op_type, op_nonzeros>::yes            DONE
  //   )
  //   ||
  //   (
  //   // operations which result in a column vector if the input is a column vector
  //   T1::is_col &&
  //     (
  //        is_same_type<op_type, op_sort>::yes
  //     || is_same_type<op_type, op_sort_default>::yes
  //     || is_same_type<op_type, op_shift>::yes
  //     || is_same_type<op_type, op_shift_default>::yes
  //     || is_same_type<op_type, op_shuffle>::yes
  //     || is_same_type<op_type, op_shuffle_default>::yes
  //     || is_same_type<op_type, op_cumsum_default>::yes
  //     || is_same_type<op_type, op_cumprod_default>::yes
  //     || is_same_type<op_type, op_flipud>::yes
  //     || is_same_type<op_type, op_fliplr>::yes
  //     || is_same_type<op_type, op_reverse>::yes
  //     || is_same_type<op_type, op_reverse_default>::yes
  //     || is_same_type<op_type, op_unique>::yes
  //     || is_same_type<op_type, op_diff_default>::yes
  //     || is_same_type<op_type, op_normalise_vec>::yes
  //     || is_same_type<op_type, op_chi2rnd>::yes
  //     )
  //   )
  //   ||
  //   (
  //   // operations which result in a column vector if the input is a row vector
  //   T1::is_row && 
  //     (
  //        is_same_type<op_type, op_strans>::yes
  //     || is_same_type<op_type, op_htrans>::yes
  //     || is_same_type<op_type, op_htrans2>::yes
  //     )
  //   )
  //   ;
  // 
  // static const bool is_xvec = \
  //   (
  //   // operations which always result in an xvec
  //      is_same_type<op_type, op_unique>::yes
  //   || is_same_type<op_type, op_cumsum>::yes
  //   || is_same_type<op_type, op_cumprod>::yes
  //   || is_same_type<op_type, op_sum>::yes
  //   || is_same_type<op_type, op_mean>::yes
  //   || is_same_type<op_type, op_median>::yes
  //   || is_same_type<op_type, op_vectorise_all>::yes
  //   || is_same_type<op_type, op_min>::yes
  //   || is_same_type<op_type, op_max>::yes
  //   || is_same_type<op_type, op_prod>::yes
  //   || is_same_type<op_type, op_range>::yes
  //   )
  //   ||
  //   (
  //   // operations which result in an xvec if the input is an xvec
  //   T1::is_xvec &&
  //     (
  //        is_same_type<op_type, op_strans>::yes
  //     || is_same_type<op_type, op_htrans>::yes
  //     || is_same_type<op_type, op_reverse>::yes
  //     || is_same_type<op_type, op_reverse_default>::yes
  //     || is_same_type<op_type, op_fliplr>::yes
  //     || is_same_type<op_type, op_flipud>::yes
  //     || is_same_type<op_type, op_sort>::yes
  //     || is_same_type<op_type, op_sort_default>::yes
  //     || is_same_type<op_type, op_shift>::yes
  //     || is_same_type<op_type, op_shift_default>::yes
  //     || is_same_type<op_type, op_shuffle>::yes
  //     || is_same_type<op_type, op_shuffle_default>::yes
  //     || is_same_type<op_type, op_diff>::yes
  //     || is_same_type<op_type, op_diff_default>::yes
  //     || is_same_type<op_type, op_normalise_vec>::yes
  //     )
  //   );
  };



//! @}
